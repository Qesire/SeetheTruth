'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=f=>fs.readFileSync(path.join(root,f),'utf8');
const shared=require('../teaching_curriculum.js');
assert.equal(shared.lessons.length,8,'mobile APK shares the PC course database');
const home=read('android_teaching.html'),policy=read('android_teaching_mode.js'),
 activity=read('android/app/src/main/java/io/github/qesire/seethetruth/TeachingActivity.java'),
 manifest=read('android/app/src/main/AndroidManifest.xml'),
 gradle=read('android/app/build.gradle'),workflow=read('.github/workflows/android-teaching-apk.yml');
const links=[...home.matchAll(/href="([^"]+)"/g)].map(m=>m[1]);
const expected=['learn.html','tradeoffs.html','paper_lab.html',
 'deepinfer_worker_lab.html','deepinfer_kv_lifecycle_lab.html',
 'deepinfer_scheduler_lab.html','deepinfer_prefill_lab.html','cache.html'];
for(const name of expected){
 assert.ok(links.some(h=>h.split('?')[0]===name),'missing learning entry '+name);
 assert.ok(fs.statSync(path.join(root,name)).isFile(),'missing bundled learning page '+name);
}
assert.match(home,/纯教学版/);
assert.match(home,/不生成 Python 程序/);
assert.match(policy,/#exportPython/);
assert.match(policy,/#pythonExportPanel/);
assert.match(policy,/#save/);
assert.match(policy,/#load/);
assert.match(policy,/button\[id\^="download"\]/);
assert.match(policy,/data-preset/);
assert.match(activity,/WebViewAssetLoader/);
assert.match(activity,/https:\/\//);
assert.match(activity,/appassets\.androidplatform\.net/);
assert.match(activity,/setAllowFileAccess\(false\)/);
assert.match(activity,/setAllowContentAccess\(false\)/);
assert.match(activity,/setJavaScriptEnabled\(true\)/);
assert.match(activity,/setDownloadListener/);
assert.match(activity,/shouldInterceptRequest/);
assert.match(activity,/blocked\(\)/);
assert.match(activity,/shouldOverrideUrlLoading/);
assert.match(activity,/teaching\.html\?edition=teaching/);
assert.match(activity,/android_teaching_mode\.js/);
assert.doesNotMatch(manifest,/<uses-permission\s+android:name="android.permission.INTERNET"/);
assert.match(manifest,/android:exported="true"/);
assert.match(gradle,/include '\*\.html'/);
assert.match(gradle,/include '\*\.js'/);
assert.match(gradle,/stageTeachingAssets/);
assert.doesNotMatch(gradle,/include '\*\.py'/);
assert.match(workflow,/:app:assembleDebug/);
assert.match(workflow,/seethetruth-android-teaching-debug-apk/);
assert.match(workflow,/unzip -t/);
const page=read('learn.html');
for(const file of [...page.matchAll(/<script[^>]*src="([^"]+)"/g)].map(m=>m[1])){
 assert.ok(fs.existsSync(path.join(root,file)),'Android missing workbench JS: '+file);
}
console.log('ANDROID_TEACHING_CONTRACTS_PASS offline entry links, no backend, no exports, guarded WebView, APK checks');
