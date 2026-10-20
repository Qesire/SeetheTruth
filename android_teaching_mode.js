/* Injected by Android TeachingActivity only; desktop features stay unchanged. */
(function(){"use strict";
if(document.documentElement.dataset.sttTeaching==="1")return;
document.documentElement.dataset.sttTeaching="1";
const css=document.createElement("style");css.id="stt-android-teaching-policy";
css.textContent=`
/* Teaching-only: no emitted files, imports, trace uploads or generated source. */
#exportPython,#exportTorch,#save,#load,#exportScope,#pythonExportPanel,
#pythonExportSource,#pythonExportHelp,#pythonExportCaption,
#traceReferenceFile,#download,#exportJson,#downloadJson,
button[id^="download"],button[id^="export"],a[download],
label[for="load"],label[for="traceReferenceFile"],
input[type=file],[data-android-hide-output] {display:none!important}
@media(max-width:680px){
 .app{display:flex!important;flex-direction:column!important;height:auto!important;min-height:0!important}
 .rail,.inspector{width:100%!important;max-height:none!important;overflow:visible!important}
 .stage{min-height:550px!important}
 .board-viewport{min-height:410px!important}
 .header{padding:10px!important}
 .head-actions{gap:5px!important}
 .playbar{flex-wrap:wrap!important}
}
`;
document.head.appendChild(css);
// Defense in depth: the WebView also disables native downloads and file pickers.
document.addEventListener("click",function(event){
 const item=event.target.closest?.("a[download],button[id^=download],button[id^=export],#save");
 if(item){event.preventDefault();event.stopImmediatePropagation();}
},true);
const p=new URLSearchParams(location.search);
const preset=p.get("mission");
if(preset && /^[a-z0-9_]{1,64}$/.test(preset)){
 const target=[...document.querySelectorAll("[data-preset]")].find(e=>e.getAttribute("data-preset")===preset);
 if(target)target.click();
}
})();