#include "etfmm/ws_client.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef _WIN32
// Load TLS roots from Windows' system certificate store rather than
// relying on OpenSSL's default verify paths (which require an
// environment-supplied cert bundle that vcpkg doesn't provide).
#include <openssl/x509.h>
#include <wincrypt.h>
#pragma comment(lib, "crypt32.lib")
#endif

namespace etfmm {

namespace {

#ifdef _WIN32
void load_windows_root_certs(boost::asio::ssl::context& ctx) {
    HCERTSTORE h = ::CertOpenSystemStoreW(0, L"ROOT");
    if (!h) {
        throw std::runtime_error("CertOpenSystemStoreW(ROOT) failed");
    }
    X509_STORE* store = ::SSL_CTX_get_cert_store(ctx.native_handle());
    PCCERT_CONTEXT cert = nullptr;
    int loaded = 0;
    while ((cert = ::CertEnumCertificatesInStore(h, cert)) != nullptr) {
        const unsigned char* p = cert->pbCertEncoded;
        X509* x509 = d2i_X509(nullptr, &p, cert->cbCertEncoded);
        if (x509) {
            // X509_STORE_add_cert dedups silently; ignore failures
            // (most likely "already in store").
            X509_STORE_add_cert(store, x509);
            X509_free(x509);
            ++loaded;
        }
    }
    ::CertCloseStore(h, 0);
    if (loaded == 0) {
        throw std::runtime_error("no Windows root certs loaded");
    }
}
#endif

}  // namespace

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace websocket = boost::beast::websocket;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

WsClient::WsClient(std::string host, std::string port, std::string target)
    : host_(std::move(host)),
      port_(std::move(port)),
      target_(std::move(target)) {}

void WsClient::stop() { stopping_.store(true, std::memory_order_release); }

void WsClient::run() {
    using namespace std::chrono_literals;
    auto backoff = 1s;
    constexpr auto kMaxBackoff = 30s;

    while (!stopping_.load(std::memory_order_acquire)) {
        try {
            run_session_();
            // Clean close: reset backoff for the next reconnect attempt.
            backoff = 1s;
        } catch (const std::exception& e) {
            // Record and back off.
            std::cerr << "[ws " << host_ << "] error: " << e.what()
                      << " (reconnecting in " << backoff.count() << "s)\n";
            reconnects_.fetch_add(1, std::memory_order_relaxed);
        }

        if (stopping_.load(std::memory_order_acquire)) break;

        // Sleep with cancellation responsiveness.
        const auto deadline = std::chrono::steady_clock::now() + backoff;
        while (std::chrono::steady_clock::now() < deadline) {
            if (stopping_.load(std::memory_order_acquire)) return;
            std::this_thread::sleep_for(100ms);
        }
        backoff = std::min(backoff * 2, kMaxBackoff);
    }
}

void WsClient::run_session_() {
    asio::io_context ioc;
    ssl::context ssl_ctx{ssl::context::tlsv12_client};
#ifdef _WIN32
    load_windows_root_certs(ssl_ctx);
#else
    ssl_ctx.set_default_verify_paths();
#endif
    ssl_ctx.set_verify_mode(ssl::verify_peer);

    tcp::resolver resolver{ioc};
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws{ioc, ssl_ctx};

    // SNI is required for many TLS endpoints (Binance among them).
    if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(),
                                  host_.c_str())) {
        throw beast::system_error{
            beast::error_code(static_cast<int>(::ERR_get_error()),
                              asio::error::get_ssl_category()),
            "Failed to set SNI hostname"};
    }

    const auto results = resolver.resolve(host_, port_);
    auto& tcp_layer = beast::get_lowest_layer(ws);
    tcp_layer.expires_after(std::chrono::seconds(15));
    tcp_layer.connect(results);

    // TLS handshake.
    tcp_layer.expires_after(std::chrono::seconds(15));
    ws.next_layer().handshake(ssl::stream_base::client);

    // WebSocket handshake. Binance accepts the host header without the
    // port suffix; setting User-Agent helps with their connection logs.
    ws.set_option(websocket::stream_base::decorator(
        [](websocket::request_type& req) {
            req.set(beast::http::field::user_agent,
                    "etf-mm-sim-cpp/0.1");
        }));
    // Disable expires_after for the WS phase so the read loop isn't
    // killed by the connect-phase timeout.
    tcp_layer.expires_never();
    ws.handshake(host_, target_);

    if (on_connected_) on_connected_();

    // Read loop.
    beast::flat_buffer buf;
    while (!stopping_.load(std::memory_order_acquire)) {
        ws.read(buf);
        const auto data = buf.data();
        const auto* p = static_cast<const char*>(data.data());
        const std::size_t n = data.size();
        if (on_text_) {
            on_text_(std::string_view(p, n));
        }
        msg_count_.fetch_add(1, std::memory_order_relaxed);
        buf.consume(n);
    }

    // Best-effort clean close.
    try {
        ws.close(websocket::close_code::normal);
    } catch (...) {
        // Ignore; the stop signal is what matters.
    }
}

}  // namespace etfmm
