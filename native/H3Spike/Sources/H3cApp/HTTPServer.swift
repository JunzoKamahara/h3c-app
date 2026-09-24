import Foundation

/// A parsed HTTP/1.1 request. Bodies are expected to be small (JSON), so
/// this is read fully into memory - there is no streaming/chunked-transfer
/// support, which APIServer's endpoints never need.
struct HTTPRequest: Sendable {
    let method: String
    let path: String
    let query: [String: String]
    let headers: [String: String]
    let body: Data
}

struct HTTPResponse: Sendable {
    var status: Int
    var statusText: String
    var headers: [String: String]
    var body: Data

    init(status: Int, statusText: String? = nil, headers: [String: String] = [:], body: Data = Data()) {
        self.status = status
        self.statusText = statusText ?? Self.defaultStatusText(status)
        self.headers = headers
        self.body = body
    }

    private static func defaultStatusText(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 202: return "Accepted"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 409: return "Conflict"
        case 500: return "Internal Server Error"
        default: return "OK"
        }
    }

    static func json(_ status: Int, _ object: Any) -> HTTPResponse {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
        return HTTPResponse(status: status, headers: ["Content-Type": "application/json"], body: data)
    }

    static func text(_ status: Int, _ text: String) -> HTTPResponse {
        HTTPResponse(status: status, headers: ["Content-Type": "text/plain; charset=utf-8"], body: Data(text.utf8))
    }

    static func error(_ status: Int, _ message: String) -> HTTPResponse {
        .json(status, ["error": message])
    }

    static func file(atPath path: String, contentType: String) -> HTTPResponse {
        guard let data = FileManager.default.contents(atPath: path) else {
            return .error(404, "file not found")
        }
        return HTTPResponse(status: 200, headers: ["Content-Type": contentType], body: data)
    }
}

/// A minimal, dependency-free HTTP/1.1 server over raw POSIX sockets -
/// deliberately not Network.framework or any third-party library, matching
/// this app's "no external dependency" approach elsewhere (native
/// AVFoundation media I/O, the plain-URLSession model downloader). Bound to
/// 127.0.0.1 only. One OS thread per connection with blocking reads/writes:
/// entirely appropriate for a local automation API that a script hits
/// occasionally, not a public-facing server under real concurrency.
final class HTTPServer {
    typealias Router = @Sendable (HTTPRequest) async -> HTTPResponse

    private let port: UInt16
    private let router: Router
    private var listenSocket: Int32 = -1
    private var running = false

    init(port: UInt16, router: @escaping Router) {
        self.port = port
        self.router = router
    }

    /// Throws a short human-readable message (not a full Error type - the
    /// only caller just wants something to print/show, never to branch on).
    func start() throws {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { throw HTTPServerError.socketFailed("socket() failed") }
        var reuse: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bindResult = withUnsafePointer(to: &address) { pointer -> Int32 in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                bind(sock, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            close(sock)
            throw HTTPServerError.socketFailed("bind() failed on port \(port) - is another process using it?")
        }
        guard listen(sock, 16) == 0 else {
            close(sock)
            throw HTTPServerError.socketFailed("listen() failed")
        }

        listenSocket = sock
        running = true
        let thread = Thread { [weak self] in self?.acceptLoop() }
        thread.name = "H3 API accept loop"
        thread.start()
    }

    func stop() {
        running = false
        if listenSocket >= 0 {
            close(listenSocket)
            listenSocket = -1
        }
    }

    private func acceptLoop() {
        while running {
            let client = accept(listenSocket, nil, nil)
            guard client >= 0 else {
                if running { continue }
                break
            }
            let thread = Thread { [weak self] in self?.handle(client: client) }
            thread.name = "H3 API connection"
            thread.start()
        }
    }

    private func handle(client: Int32) {
        defer { close(client) }
        guard let request = Self.readRequest(socket: client) else { return }

        let semaphore = DispatchSemaphore(value: 0)
        let router = self.router
        let box = ResponseBox()
        Task {
            box.response = await router(request)
            semaphore.signal()
        }
        semaphore.wait()
        Self.write(box.response ?? .error(500, "no response"), to: client)
    }

    private static func readRequest(socket: Int32) -> HTTPRequest? {
        var buffer = Data()
        let separator = Data("\r\n\r\n".utf8)
        var chunk = [UInt8](repeating: 0, count: 65536)

        while buffer.range(of: separator) == nil {
            let n = recv(socket, &chunk, chunk.count, 0)
            guard n > 0 else { return nil }
            buffer.append(contentsOf: chunk[0 ..< n])
            if buffer.count > 1 << 20 { return nil } // headers this large mean something is wrong
        }

        guard let headerEndRange = buffer.range(of: separator),
              let headText = String(data: buffer[..<headerEndRange.lowerBound], encoding: .utf8) else { return nil }
        var bodyBuffer = buffer[headerEndRange.upperBound...]

        let lines = headText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let requestParts = requestLine.split(separator: " ", maxSplits: 2)
        guard requestParts.count >= 2 else { return nil }
        let method = String(requestParts[0])
        let rawPath = String(requestParts[1])

        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[line.startIndex ..< colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[key] = value
        }

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        while bodyBuffer.count < contentLength {
            let n = recv(socket, &chunk, min(chunk.count, contentLength - bodyBuffer.count), 0)
            guard n > 0 else { break }
            bodyBuffer.append(contentsOf: chunk[0 ..< n])
        }

        let (path, query) = Self.splitPathAndQuery(rawPath)
        return HTTPRequest(method: method, path: path, query: query, headers: headers, body: Data(bodyBuffer))
    }

    private static func splitPathAndQuery(_ rawPath: String) -> (path: String, query: [String: String]) {
        guard let questionIndex = rawPath.firstIndex(of: "?") else {
            return (rawPath.removingPercentEncoding ?? rawPath, [:])
        }
        let path = String(rawPath[rawPath.startIndex ..< questionIndex])
        let queryString = String(rawPath[rawPath.index(after: questionIndex)...])
        var query: [String: String] = [:]
        for pair in queryString.split(separator: "&") {
            let keyValue = pair.split(separator: "=", maxSplits: 1)
            guard let key = keyValue.first else { continue }
            let value = keyValue.count > 1 ? String(keyValue[1]) : ""
            query[String(key).removingPercentEncoding ?? String(key)] = value.removingPercentEncoding ?? value
        }
        return (path.removingPercentEncoding ?? path, query)
    }

    private static func write(_ response: HTTPResponse, to socket: Int32) {
        var headers = response.headers
        headers["Content-Length"] = "\(response.body.count)"
        headers["Connection"] = "close"
        var head = "HTTP/1.1 \(response.status) \(response.statusText)\r\n"
        for (key, value) in headers { head += "\(key): \(value)\r\n" }
        head += "\r\n"

        var payload = Data(head.utf8)
        payload.append(response.body)
        payload.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard var pointer = raw.baseAddress else { return }
            var remaining = raw.count
            while remaining > 0 {
                let n = send(socket, pointer, remaining, 0)
                guard n > 0 else { break }
                pointer += n
                remaining -= n
            }
        }
    }
}

/// A plain mutable box to hand a response back out of the `Task` in
/// `handle(client:)` - simpler than juggling `nonisolated(unsafe)` on a
/// local `var` just to satisfy the @Sendable closure capture.
private final class ResponseBox: @unchecked Sendable {
    var response: HTTPResponse?
}

enum HTTPServerError: Error, CustomStringConvertible {
    case socketFailed(String)
    var description: String {
        switch self {
        case .socketFailed(let message): return message
        }
    }
}

