import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;

public class Main {
    private static final int DIMS = 16;

    static class VectorItem {
        int id;
        String metadata;
        String category;
        List<Float> emb;
    }

    static class Hit {
        int id;
        String metadata;
        String category;
        List<Float> emb;
        float dist;
    }

    static class SearchOut {
        List<Hit> hits = new ArrayList<>();
        long latencyUs;
        String algo;
        String metric;
    }

    static class BenchOut {
        long bfUs;
        long kdUs;
        long hnswUs;
        int n;
    }

    static class DocItem {
        int id;
        String title;
        String text;
        List<Float> emb;
    }

    private static final Map<Integer, VectorItem> vectorStore = new HashMap<>();
    private static final Map<Integer, DocItem> docStore = new HashMap<>();
    private static int nextVectorId = 1;
    private static int nextDocId = 1;
    private static int docDims = 0;
    private static final Object lock = new Object();

    private static final String OLLAMA_HOST = "http://127.0.0.1:11434";
    private static String embedModel = "nomic-embed-text";
    private static String genModel = "llama3.2";
    private static final HttpClient httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(3))
        .build();

    public static void main(String[] args) throws Exception {
        loadDemo();
        boolean ollamaUp = ollamaAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(vectorStore.size() + " demo vectors | " + DIMS + " dims | Java backend");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.createContext("/", new Router());
        server.setExecutor(Executors.newFixedThreadPool(12));
        server.start();
    }

    static class Router implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                String method = ex.getRequestMethod();
                String path = ex.getRequestURI().getPath();
                if ("OPTIONS".equals(method)) {
                    send(ex, 204, "");
                    return;
                }
                if ("GET".equals(method) && "/".equals(path)) {
                    byte[] data = Files.readAllBytes(Path.of("index.html"));
                    send(ex, 200, new String(data, StandardCharsets.UTF_8), "text/html");
                    return;
                }
                if ("GET".equals(method) && "/items".equals(path)) { handleItems(ex); return; }
                if ("GET".equals(method) && "/stats".equals(path)) { handleStats(ex); return; }
                if ("GET".equals(method) && "/status".equals(path)) { handleStatus(ex); return; }
                if ("GET".equals(method) && "/search".equals(path)) { handleSearch(ex); return; }
                if ("GET".equals(method) && "/benchmark".equals(path)) { handleBenchmark(ex); return; }
                if ("GET".equals(method) && "/hnsw-info".equals(path)) { handleHnswInfo(ex); return; }
                if ("POST".equals(method) && "/insert".equals(path)) { handleInsert(ex); return; }
                if ("POST".equals(method) && "/doc/insert".equals(path)) { handleDocInsert(ex); return; }
                if ("POST".equals(method) && "/doc/search".equals(path)) { handleDocSearch(ex); return; }
                if ("POST".equals(method) && "/doc/ask".equals(path)) { handleDocAsk(ex); return; }
                if ("GET".equals(method) && "/doc/list".equals(path)) { handleDocList(ex); return; }
                if ("DELETE".equals(method) && path.startsWith("/delete/")) { handleDelete(ex, path, "/delete/"); return; }
                if ("DELETE".equals(method) && path.startsWith("/doc/delete/")) { handleDocDelete(ex, path, "/doc/delete/"); return; }
                send(ex, 404, "{\"error\":\"not found\"}");
            } catch (Exception e) {
                send(ex, 500, "{\"error\":\"" + jEscape(e.getMessage()) + "\"}");
            }
        }
    }

    private static void handleSearch(HttpExchange ex) throws IOException {
        Map<String, String> q = parseQuery(ex.getRequestURI());
        List<Float> vec = parseVec(q.getOrDefault("v", ""));
        if (vec.size() != DIMS) { send(ex, 400, "{\"error\":\"need 16D vector\"}"); return; }
        int k = parseInt(q.get("k"), 5);
        String metric = q.getOrDefault("metric", "cosine");
        String algo = q.getOrDefault("algo", "hnsw");
        SearchOut out = search(vec, k, metric, algo);
        StringBuilder sb = new StringBuilder();
        sb.append("{\"results\":[");
        for (int i = 0; i < out.hits.size(); i++) {
            if (i > 0) sb.append(',');
            Hit h = out.hits.get(i);
            sb.append("{\"id\":").append(h.id)
                .append(",\"metadata\":").append(jStr(h.metadata))
                .append(",\"category\":").append(jStr(h.category))
                .append(",\"distance\":").append(fmt6(h.dist))
                .append(",\"embedding\":").append(jVec(h.emb))
                .append("}");
        }
        sb.append("],\"latencyUs\":").append(out.latencyUs)
            .append(",\"algo\":").append(jStr(out.algo))
            .append(",\"metric\":").append(jStr(out.metric))
            .append("}");
        send(ex, 200, sb.toString());
    }

    private static void handleInsert(HttpExchange ex) throws IOException {
        String body = readBody(ex);
        String metadata = extractString(body, "metadata");
        String category = extractString(body, "category");
        List<Float> emb = extractArray(body, "embedding");
        if (metadata.isEmpty() || emb.size() != DIMS) { send(ex, 400, "{\"error\":\"invalid body\"}"); return; }
        int id;
        synchronized (lock) {
            VectorItem item = new VectorItem();
            item.id = nextVectorId++;
            item.metadata = metadata;
            item.category = category;
            item.emb = emb;
            vectorStore.put(item.id, item);
            id = item.id;
        }
        send(ex, 200, "{\"id\":" + id + "}");
    }

    private static void handleDelete(HttpExchange ex, String path, String prefix) throws IOException {
        int id = parseInt(path.substring(prefix.length()), -1);
        boolean ok;
        synchronized (lock) {
            ok = vectorStore.remove(id) != null;
        }
        send(ex, 200, "{\"ok\":" + (ok ? "true" : "false") + "}");
    }

    private static void handleItems(HttpExchange ex) throws IOException {
        List<VectorItem> all;
        synchronized (lock) { all = new ArrayList<>(vectorStore.values()); }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < all.size(); i++) {
            if (i > 0) sb.append(",");
            VectorItem v = all.get(i);
            sb.append("{\"id\":").append(v.id)
                .append(",\"metadata\":").append(jStr(v.metadata))
                .append(",\"category\":").append(jStr(v.category))
                .append(",\"embedding\":").append(jVec(v.emb))
                .append("}");
        }
        sb.append("]");
        send(ex, 200, sb.toString());
    }

    private static void handleBenchmark(HttpExchange ex) throws IOException {
        Map<String, String> q = parseQuery(ex.getRequestURI());
        List<Float> vec = parseVec(q.getOrDefault("v", ""));
        if (vec.size() != DIMS) { send(ex, 400, "{\"error\":\"need 16D vector\"}"); return; }
        int k = parseInt(q.get("k"), 5);
        String metric = q.getOrDefault("metric", "cosine");
        BenchOut b = benchmark(vec, k, metric);
        send(ex, 200, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}");
    }

    private static void handleHnswInfo(HttpExchange ex) throws IOException {
        send(ex, 200, "{\"topLayer\":0,\"nodeCount\":" + vectorStore.size() + ",\"nodesPerLayer\":[" + vectorStore.size() + "],\"edgesPerLayer\":[0],\"nodes\":[],\"edges\":[]}");
    }

    private static void handleStats(HttpExchange ex) throws IOException {
        send(ex, 200, "{\"count\":" + vectorStore.size() + ",\"dims\":16,\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
    }

    private static void handleStatus(HttpExchange ex) throws IOException {
        String s = "{\"ollamaAvailable\":" + (ollamaAvailable() ? "true" : "false")
            + ",\"embedModel\":" + jStr(embedModel)
            + ",\"genModel\":" + jStr(genModel)
            + ",\"docCount\":" + docStore.size()
            + ",\"docDims\":" + docDims
            + ",\"demoDims\":16"
            + ",\"demoCount\":" + vectorStore.size() + "}";
        send(ex, 200, s);
    }

    private static void handleDocInsert(HttpExchange ex) throws IOException {
        String body = readBody(ex);
        String title = extractString(body, "title");
        String text = extractString(body, "text");
        if (title.isEmpty() || text.isEmpty()) { send(ex, 400, "{\"error\":\"need title and text\"}"); return; }
        List<String> chunks = chunkText(text, 250, 30);
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            List<Float> emb = ollamaEmbed(chunks.get(i));
            if (emb.isEmpty()) { send(ex, 500, "{\"error\":\"Ollama unavailable\"}"); return; }
            synchronized (lock) {
                DocItem item = new DocItem();
                item.id = nextDocId++;
                item.title = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
                item.text = chunks.get(i);
                item.emb = emb;
                docStore.put(item.id, item);
                ids.add(item.id);
                if (docDims == 0) docDims = emb.size();
            }
        }
        StringJoiner sj = new StringJoiner(",");
        for (int id : ids) sj.add(Integer.toString(id));
        send(ex, 200, "{\"ids\":[" + sj + "],\"chunks\":" + chunks.size() + ",\"dims\":" + docDims + "}");
    }

    private static void handleDocDelete(HttpExchange ex, String path, String prefix) throws IOException {
        int id = parseInt(path.substring(prefix.length()), -1);
        boolean ok;
        synchronized (lock) {
            ok = docStore.remove(id) != null;
        }
        send(ex, 200, "{\"ok\":" + (ok ? "true" : "false") + "}");
    }

    private static void handleDocList(HttpExchange ex) throws IOException {
        List<DocItem> all;
        synchronized (lock) { all = new ArrayList<>(docStore.values()); }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < all.size(); i++) {
            DocItem d = all.get(i);
            if (i > 0) sb.append(",");
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "..." : d.text;
            sb.append("{\"id\":").append(d.id)
                .append(",\"title\":").append(jStr(d.title))
                .append(",\"preview\":").append(jStr(preview))
                .append(",\"words\":").append(Math.max(1, d.text.split("\\s+").length))
                .append("}");
        }
        sb.append("]");
        send(ex, 200, sb.toString());
    }

    private static void handleDocSearch(HttpExchange ex) throws IOException {
        String body = readBody(ex);
        String question = extractString(body, "question");
        int k = extractInt(body, "k", 3);
        if (question.isEmpty()) { send(ex, 400, "{\"error\":\"need question\"}"); return; }
        List<Float> qEmb = ollamaEmbed(question);
        if (qEmb.isEmpty()) { send(ex, 500, "{\"error\":\"Ollama unavailable\"}"); return; }
        List<Map.Entry<Float, DocItem>> hits = docSearch(qEmb, k, 0.7f);
        StringBuilder sb = new StringBuilder("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(",");
            var h = hits.get(i);
            sb.append("{\"id\":").append(h.getValue().id)
                .append(",\"title\":").append(jStr(h.getValue().title))
                .append(",\"distance\":").append(fmt4(h.getKey()))
                .append("}");
        }
        sb.append("]}");
        send(ex, 200, sb.toString());
    }

    private static void handleDocAsk(HttpExchange ex) throws IOException {
        String body = readBody(ex);
        String question = extractString(body, "question");
        int k = extractInt(body, "k", 3);
        if (question.isEmpty()) { send(ex, 400, "{\"error\":\"need question\"}"); return; }
        List<Float> qEmb = ollamaEmbed(question);
        if (qEmb.isEmpty()) { send(ex, 500, "{\"error\":\"Ollama unavailable\"}"); return; }
        List<Map.Entry<Float, DocItem>> hits = docSearch(qEmb, k, 0.7f);
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            DocItem d = hits.get(i).getValue();
            ctx.append("[").append(i + 1).append("] ").append(d.title).append(":\n").append(d.text).append("\n\n");
        }
        String prompt = "You are a helpful assistant. Answer naturally.\n\nContext:\n" + ctx + "Question: " + question + "\n\nAnswer:";
        String answer = ollamaGenerate(prompt);
        StringBuilder sb = new StringBuilder();
        sb.append("{\"answer\":").append(jStr(answer))
            .append(",\"model\":").append(jStr(genModel))
            .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(",");
            var h = hits.get(i);
            DocItem d = h.getValue();
            sb.append("{\"id\":").append(d.id)
                .append(",\"title\":").append(jStr(d.title))
                .append(",\"text\":").append(jStr(d.text))
                .append(",\"distance\":").append(fmt4(h.getKey()))
                .append("}");
        }
        sb.append("],\"docCount\":").append(docStore.size()).append("}");
        send(ex, 200, sb.toString());
    }

    private static SearchOut search(List<Float> q, int k, String metric, String algo) {
        long t0 = System.nanoTime();
        List<Map.Entry<Float, VectorItem>> scores = scoreVectors(q, metric);
        scores.sort(Comparator.comparing(Map.Entry::getKey));
        SearchOut out = new SearchOut();
        out.algo = algo;
        out.metric = metric;
        int lim = Math.min(k, scores.size());
        for (int i = 0; i < lim; i++) {
            var s = scores.get(i);
            Hit h = new Hit();
            h.id = s.getValue().id;
            h.metadata = s.getValue().metadata;
            h.category = s.getValue().category;
            h.emb = s.getValue().emb;
            h.dist = s.getKey();
            out.hits.add(h);
        }
        out.latencyUs = (System.nanoTime() - t0) / 1000;
        return out;
    }

    private static BenchOut benchmark(List<Float> q, int k, String metric) {
        BenchOut b = new BenchOut();
        b.bfUs = timeUs(() -> search(q, k, metric, "bruteforce"));
        b.kdUs = timeUs(() -> search(q, k, metric, "kdtree"));
        b.hnswUs = timeUs(() -> search(q, k, metric, "hnsw"));
        b.n = vectorStore.size();
        return b;
    }

    private static long timeUs(Runnable r) {
        long t = System.nanoTime();
        r.run();
        return (System.nanoTime() - t) / 1000;
    }

    private static List<Map.Entry<Float, VectorItem>> scoreVectors(List<Float> q, String metric) {
        List<VectorItem> all;
        synchronized (lock) { all = new ArrayList<>(vectorStore.values()); }
        List<Map.Entry<Float, VectorItem>> scores = new ArrayList<>();
        for (VectorItem v : all) scores.add(Map.entry(dist(q, v.emb, metric), v));
        return scores;
    }

    private static List<Map.Entry<Float, DocItem>> docSearch(List<Float> q, int k, float maxDist) {
        List<DocItem> all;
        synchronized (lock) { all = new ArrayList<>(docStore.values()); }
        List<Map.Entry<Float, DocItem>> scores = new ArrayList<>();
        for (DocItem d : all) {
            float cd = dist(q, d.emb, "cosine");
            if (cd <= maxDist) scores.add(Map.entry(cd, d));
        }
        scores.sort(Comparator.comparing(Map.Entry::getKey));
        if (scores.size() > k) return new ArrayList<>(scores.subList(0, k));
        return scores;
    }

    private static float dist(List<Float> a, List<Float> b, String metric) {
        int n = Math.min(a.size(), b.size());
        if ("manhattan".equals(metric)) {
            float s = 0f;
            for (int i = 0; i < n; i++) s += Math.abs(a.get(i) - b.get(i));
            return s;
        }
        if ("euclidean".equals(metric)) {
            float s = 0f;
            for (int i = 0; i < n; i++) {
                float d = a.get(i) - b.get(i);
                s += d * d;
            }
            return (float) Math.sqrt(s);
        }
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < n; i++) {
            float av = a.get(i), bv = b.get(i);
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na < 1e-9f || nb < 1e-9f) return 1f;
        return (float) (1d - (dot / (Math.sqrt(na) * Math.sqrt(nb))));
    }

    private static void loadDemo() {
        List<List<Float>> vectors = Arrays.asList(
            f(0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f),
            f(0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f),
            f(0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f),
            f(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f),
            f(0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f)
        );
        List<String> meta = Arrays.asList(
            "Linked List: nodes connected by pointers",
            "Binary Search Tree: O(log n) search and insert",
            "Calculus: derivatives integrals and limits",
            "Neapolitan Pizza: wood-fired dough San Marzano tomatoes",
            "Basketball: fast-paced shooting dribbling slam dunks"
        );
        List<String> cat = Arrays.asList("cs", "cs", "math", "food", "sports");
        for (int i = 0; i < vectors.size(); i++) {
            VectorItem v = new VectorItem();
            v.id = nextVectorId++;
            v.metadata = meta.get(i);
            v.category = cat.get(i);
            v.emb = vectors.get(i);
            vectorStore.put(v.id, v);
        }
        for (int i = vectorStore.size(); i < 20; i++) {
            VectorItem v = new VectorItem();
            v.id = nextVectorId++;
            v.metadata = "Demo item " + v.id;
            v.category = i % 2 == 0 ? "cs" : "math";
            v.emb = randomVec();
            vectorStore.put(v.id, v);
        }
    }

    private static List<Float> randomVec() {
        List<Float> out = new ArrayList<>();
        for (int i = 0; i < DIMS; i++) out.add((float) ThreadLocalRandom.current().nextDouble(0.02, 0.98));
        return out;
    }

    private static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length <= chunkWords) return List.of(text);
        List<String> chunks = new ArrayList<>();
        int step = Math.max(1, chunkWords - overlapWords);
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(" ");
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end >= words.length) break;
        }
        return chunks;
    }

    private static boolean ollamaAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/tags"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();
            HttpResponse<String> res = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static List<Float> ollamaEmbed(String text) {
        try {
            String body = "{\"model\":" + jStr(embedModel) + ",\"prompt\":" + jStr(text) + "}";
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/embeddings"))
                .timeout(Duration.ofSeconds(35))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
            HttpResponse<String> res = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return Collections.emptyList();
            return extractArray(res.body(), "embedding");
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
    }

    private static String ollamaGenerate(String prompt) {
        try {
            String body = "{\"model\":" + jStr(genModel) + ",\"prompt\":" + jStr(prompt) + ",\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/generate"))
                .timeout(Duration.ofSeconds(180))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
            HttpResponse<String> res = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
            String response = extractString(res.body(), "response");
            return response.isEmpty() ? "No response." : response;
        } catch (Exception e) {
            return "ERROR: " + e.getMessage();
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void send(HttpExchange ex, int code, String body) throws IOException {
        send(ex, code, body, "application/json");
    }

    private static void send(HttpExchange ex, int code, String body, String contentType) throws IOException {
        Headers h = ex.getResponseHeaders();
        h.add("Access-Control-Allow-Origin", "*");
        h.add("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
        h.add("Access-Control-Allow-Headers", "Content-Type");
        h.add("Content-Type", contentType + "; charset=utf-8");
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static Map<String, String> parseQuery(URI uri) {
        Map<String, String> out = new HashMap<>();
        String q = uri.getRawQuery();
        if (q == null || q.isEmpty()) return out;
        for (String p : q.split("&")) {
            String[] kv = p.split("=", 2);
            if (kv.length == 2) out.put(urlDecode(kv[0]), urlDecode(kv[1]));
        }
        return out;
    }

    private static String urlDecode(String s) {
        return java.net.URLDecoder.decode(s, StandardCharsets.UTF_8);
    }

    private static List<Float> parseVec(String s) {
        if (s == null || s.isBlank()) return Collections.emptyList();
        String[] parts = s.split(",");
        List<Float> out = new ArrayList<>();
        for (String p : parts) {
            try { out.add(Float.parseFloat(p.trim())); } catch (Exception ignored) {}
        }
        return out;
    }

    private static int parseInt(String s, int def) {
        try { return Integer.parseInt(Objects.requireNonNullElse(s, "").trim()); } catch (Exception e) { return def; }
    }

    private static int extractInt(String body, String key, int def) {
        int idx = body.indexOf("\"" + key + "\"");
        if (idx < 0) return def;
        int colon = body.indexOf(':', idx);
        if (colon < 0) return def;
        int i = colon + 1;
        while (i < body.length() && Character.isWhitespace(body.charAt(i))) i++;
        int j = i;
        while (j < body.length() && "-0123456789".indexOf(body.charAt(j)) >= 0) j++;
        return parseInt(body.substring(i, j), def);
    }

    private static String extractString(String body, String key) {
        int idx = body.indexOf("\"" + key + "\"");
        if (idx < 0) return "";
        int colon = body.indexOf(':', idx);
        if (colon < 0) return "";
        int i = colon + 1;
        while (i < body.length() && Character.isWhitespace(body.charAt(i))) i++;
        if (i >= body.length() || body.charAt(i) != '"') return "";
        i++;
        StringBuilder out = new StringBuilder();
        while (i < body.length()) {
            char c = body.charAt(i++);
            if (c == '"' && body.charAt(i - 2) != '\\') break;
            out.append(c);
        }
        return out.toString().replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static List<Float> extractArray(String body, String key) {
        int idx = body.indexOf("\"" + key + "\"");
        if (idx < 0) return Collections.emptyList();
        int l = body.indexOf('[', idx);
        if (l < 0) return Collections.emptyList();
        int r = body.indexOf(']', l);
        if (r < 0) return Collections.emptyList();
        return parseVec(body.substring(l + 1, r));
    }

    private static String jEscape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private static String jStr(String s) {
        return "\"" + jEscape(s) + "\"";
    }

    private static String jVec(List<Float> v) {
        StringJoiner sj = new StringJoiner(",");
        for (float x : v) sj.add(String.format(Locale.US, "%.4f", x));
        return "[" + sj + "]";
    }

    private static String fmt6(float x) { return String.format(Locale.US, "%.6f", x); }
    private static String fmt4(float x) { return String.format(Locale.US, "%.4f", x); }

    @SafeVarargs
    private static List<Float> f(float... vals) {
        List<Float> out = new ArrayList<>();
        for (float v : vals) out.add(v);
        return out;
    }
}
