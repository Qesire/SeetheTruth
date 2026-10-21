package io.github.qesire.seethetruth;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.CookieManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.webkit.WebViewAssetLoader;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;

/**
 * A deliberately offline, teaching-only wrapper around the existing
 * browser workbench. No JavascriptInterface, file picker, uploads, server
 * process, background downloads, or model inference service.
 */
public final class TeachingActivity extends Activity {
    private static final String HOST = "appassets.androidplatform.net";
    private static final String BASE = "https://" + HOST + "/assets/";
    private WebView webView;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        webView = new WebView(this);
        webView.setBackgroundColor(0xFF071321);
        setContentView(webView);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setAllowFileAccessFromFileURLs(false);
        settings.setAllowUniversalAccessFromFileURLs(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
        settings.setSupportZoom(true);
        settings.setSafeBrowsingEnabled(true);
        CookieManager.getInstance().setAcceptCookie(false);
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, false);
        webView.setWebChromeClient(new WebChromeClient());
        // No WebChromeClient.onShowFileChooser: cannot upload or import files.
        webView.setDownloadListener((url, agent, disposition, mime, size) -> {
            // Educational edition intentionally does not produce files.
        });
        WebViewAssetLoader assets = new WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
            .build();
        webView.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view,
                                                                        WebResourceRequest request) {
                Uri uri = request.getUrl();
                if ("https".equals(uri.getScheme()) && HOST.equals(uri.getHost())
                    && uri.getPath() != null && uri.getPath().startsWith("/assets/")) {
                    WebResourceResponse response = assets.shouldInterceptRequest(uri);
                    if (response != null) return response;
                }
                return blocked();
            }

            @Override public boolean shouldOverrideUrlLoading(WebView view,
                                                                WebResourceRequest request) {
                Uri url = request.getUrl();
                if ("https".equals(url.getScheme()) && HOST.equals(url.getHost())
                    && url.getPath() != null && url.getPath().startsWith("/assets/")) {
                    return false;
                }
                // Paper references can be opened in the user's browser by
                // explicit gestures; this offline app itself has no network permission.
                if (request.isForMainFrame() && request.hasGesture() &&
                    ("https".equals(url.getScheme()) || "http".equals(url.getScheme()))) {
                    Intent intent = new Intent(Intent.ACTION_VIEW, url);
                    try { startActivity(intent); } catch (Exception ignored) {}
                }
                return true;
            }

            @Override public void onPageFinished(WebView view, String url) {
                if (url.startsWith(BASE)) {
                    view.evaluateJavascript(
                        "(function(){if(window.__seethetruthMobileLoaded)return;" +
                        "window.__seethetruthMobileLoaded=true;" +
                        "var s=document.createElement('script');" +
                        "s.src='/assets/android_teaching_mode.js';document.head.appendChild(s)})()",
                        null);
                }
            }
        });
        String launch = BASE + "teaching.html?edition=teaching";
        if (state != null) {
            webView.restoreState(state);
        } else {
            webView.loadUrl(launch);
        }
    }

    private static WebResourceResponse blocked() {
        return new WebResourceResponse("text/plain", "UTF-8", 403,
            "Blocked by offline teaching edition", Collections.emptyMap(),
            new ByteArrayInputStream(new byte[0]));
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        webView.saveState(state);
        super.onSaveInstanceState(state);
    }

    @Override public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override protected void onDestroy() {
        if (webView != null) {
            webView.stopLoading();
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }
}
