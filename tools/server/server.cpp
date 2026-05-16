#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

// 异步控制信号灯（保持原子性与线程安全）
static std::atomic<bool> g_should_switch_model{false};
static std::string g_next_model_path = "";
static std::string g_next_model_alias = "";

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// 【优化 1】：极度加固的异常包装器，防止异常解析 JSON 失败导致的客户端死锁
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            // 终极硬编码兜底：如果上面解析 JSON 发生次生灾害，直接返回硬编码的合规 JSON 字符串，绝不外泄异常
            SRV_ERR("Secondary exception in ex_wrapper: %s | original: %s\n", e.what(), message.c_str());
            res->status = 500;
            res->data = "{\"error\":{\"code\":500,\"message\":\"Internal Server Error (JSON formatting failed)\",\"type\":\"server_error\"}}";
        } catch (...) {
            res->status = 500;
            res->data = "{\"error\":{\"code\":500,\"message\":\"Unknown fatal server error\",\"type\":\"server_error\"}}";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params);

    if (params.embedding && params.n_batch > params.n_ubatch) {
        SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
        SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    server_context ctx_server;
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    server_routes routes(params, ctx_server);
    server_tools tools;

    bool is_router_server = params.model.path.empty();
    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health));
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions));
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages));
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens));
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings));
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    ctx_http.register_gcp_compat();

    // 自定义单端口模型热切换接口
    if (!is_router_server) {
        ctx_http.post("/v1/custom/switch_model", ex_wrapper([&params](const server_http_req & req) -> server_http_res_ptr {
            auto res = std::make_unique<server_http_res>();
            try {
                json body = json::parse(req.body);
                if (!body.contains("model_path")) {
                    res->status = 400;
                    res->data = "{\"error\": \"Missing 'model_path' in request body\"}";
                    return res;
                }
                
                g_next_model_path = body["model_path"];
                g_next_model_alias = body.value("model_alias", "");
                g_should_switch_model.store(true);

                res->status = 200;
                res->data = "{\"status\": \"processing\", \"message\": \"Model switch signal received.\"}";
            } catch (const std::exception & e) {
                res->status = 500;
                res->data = "{\"error\": \"" + std::string(e.what()) + "\"}";
            }
            return res;
        }));
    }

    if (params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");
        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            return 1;
        }
        ctx_http.is_ready.store(true);
        shutdown_handler = [&](int) { ctx_http.stop(); };

    } else {
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            return 1;
        }

        SRV_INF("%s", "loading model\n");
        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) { ctx_http.thread.join(); }
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);
        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) { ctx_server.terminate(); };
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        if (ctx_http.thread.joinable()) { ctx_http.thread.join(); }
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread = server_models::setup_child_server(shutdown_handler, model_info);
        }

        // 【优化 2】：看门狗生命周期绑定。当 http 服务停止或退出时，看门狗能安全中止，防止常驻内存
        std::thread reload_watchdog([&ctx_server, &ctx_http]() {
            while (ctx_http.is_ready.load()) {
                if (g_should_switch_model.load()) {
                    ctx_server.terminate(); 
                    // 挂起看门狗，直到主线程把信号消耗完毕，防止重复调用 terminate 造成状态错乱
                    while (g_should_switch_model.load() && ctx_http.is_ready.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        reload_watchdog.detach();

        // 【优化 3】：重构事件重载循环（前置信号检查时序，防止信号漏掉）
        while (true) {
            // 在进入阻塞的 start_loop 之前，**优先检查**是否有未处理的切模型请求
            if (g_should_switch_model.load()) {
                SRV_INF("%s","Hot-reload triggered pre-loop! Unloading old inference system...\n");
                
                // 彻底中断当前状态并显式触发资源清理
                ctx_server.terminate();

                params.model.path = g_next_model_path;
                if (!g_next_model_alias.empty()) {
                    params.model_alias = { g_next_model_alias };
                } else {
                    params.model_alias = {};
                }

                SRV_INF("%s","Hot-reload! Loading new model weights from: %s\n", params.model.path.c_str());

                if (!ctx_server.load_model(params)) {
                    SRV_ERR("Hot-reload CRITICAL ERROR: Failed to load model %s\n", params.model.path.c_str());
                    break; 
                }

                routes.update_meta(ctx_server);
                g_should_switch_model.store(false); // 消费完信号，关灯
                SRV_INF("%s","Hot-reload pre-check complete. Starting service safely!\n");
            }

            // 此处会阻塞主线程，开始正常响应 /v1/chat/completions
            ctx_server.start_loop();

            // 当从 start_loop 阻塞中醒来时，再次检查是否是因为看门狗触发了换模信号
            if (g_should_switch_model.load()) {
                SRV_INF("%s","Hot-reload triggered post-loop! Unloading old inference system...\n");
                ctx_server.terminate();

                params.model.path = g_next_model_path;
                if (!g_next_model_alias.empty()) {
                    params.model_alias = { g_next_model_alias };
                } else {
                    params.model_alias = {};
                }

                SRV_INF("Hot-reload! Loading new model weights from: %s\n", params.model.path.c_str());

                if (!ctx_server.load_model(params)) {
                    SRV_ERR("Hot-reload CRITICAL ERROR: Failed to load model %s\n", params.model.path.c_str());
                    break; 
                }

                routes.update_meta(ctx_server);
                g_should_switch_model.store(false); 
                SRV_INF("%s","Hot-reload post-check complete. Resuming service loop!\n");
                
                continue; // 带着新模型重新杀回循环头部
            }

            // 真正的系统退出（非换模引起的退出，例如 Ctrl+C 终止信号）
            break;
        }

        // 规范清理
        clean_up();
        if (ctx_http.thread.joinable()) { ctx_http.thread.join(); }
        if (monitor_thread.joinable()) { monitor_thread.join(); }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
