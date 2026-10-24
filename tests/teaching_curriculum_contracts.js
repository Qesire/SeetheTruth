'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const C=require('../teaching_curriculum.js');
assert.equal(C.lessons.length,8,'all eight shared explanatory tracks must be present');
assert.equal(new Set(C.lessons.map(x=>x.id)).size,C.lessons.length);
for(const lesson of C.lessons){
 assert.ok(fs.existsSync(path.join(root,lesson.href)),lesson.href+' must be a real shared PC/Android lesson');
 for(const key of ['theory','why','check','answer'])assert.ok(lesson[key].length>28,lesson.id+' incomplete '+key);
 assert.ok(/^([a-z0-9_-]+\.html)$/.test(lesson.href));
}
const text=file=>fs.readFileSync(path.join(root,file),'utf8');
const hub=text('teaching.html'),studio=text('learn.html'),gate=text('teaching_edition.js');
const gradle=text('android/app/build.gradle');
const activity=text('android/app/src/main/java/io/github/qesire/seethetruth/TeachingActivity.java');
const manifest=text('android/app/src/main/AndroidManifest.xml');
assert.ok(hub.includes('src="teaching_curriculum.js"')&&hub.includes('src="teaching_hub.js"'));
assert.ok(hub.includes('src="deepinfer_prefill_policy.js"'),'both platforms must load the same actual vote computation');
assert.ok(text('teaching_hub.js').includes('Core.executeLab("deepinfer_prefill","vote"'),'mechanism output must come from the actual source-derived domain adapter');
assert.ok(studio.includes('href="teaching.html"'),'PC must link shared teaching entry');
assert.ok(studio.includes('src="teaching_edition.js"'),'teaching edition controls supported on desktop/mobile');
assert.ok(studio.includes('mobile_teaching.css'),'same math workbench adapts to small screens');
assert.ok(gate.includes('exportPython')&&gate.includes('save'),'APK teaching should not present artifact outputs');
assert.ok(!hub.includes('id="exportPython"')&&!hub.includes('id="save"'));
assert.ok(gradle.includes("teaching-assets.txt")&&!gradle.includes("include '*.js'"),
 'Android staging must use the audited same-source list, never root JS/HTML globs');
assert.ok(activity.includes('WebViewAssetLoader')&&activity.includes('setAllowFileAccess(false)'));
assert.ok(activity.includes('setAllowUniversalAccessFromFileURLs(false)'));
assert.ok(activity.includes('appassets.androidplatform.net'));
assert.ok(activity.includes('teaching.html?edition=teaching'));
assert.ok(!activity.includes('addJavascriptInterface('));
assert.ok(!manifest.includes('android.permission.INTERNET'),'offline education app should request no Internet permission');
assert.ok(!activity.includes('addJavascriptInterface('),'no native JS execution bridge');
console.log('TEACHING_CROSSPLATFORM_PASS shared course content, PC entry, offline Android shell and export-free mode');
