/* Attach the shared typed lesson contract to a dedicated teaching laboratory.
 * The lab keeps its actual domain engine; no synthetic graph impersonation.
 */
(function(root){"use strict";
 const script=typeof document!=="undefined"?document.currentScript:null;
 const id=script?.getAttribute("data-experience");
 const core=root.SeetheTruthUnifiedCore;
 if(!id||!core)throw Error("missing unified core / specialized experience identity");
 const contract=core.inspectLab(id);
 if(contract.href!==location.pathname.split("/").at(-1))
  throw Error("specialized experience route mismatch "+id);
 root.SeetheTruthSpecializedLesson=Object.freeze({
  ...contract,
  execute:(operation,input)=>core.executeLab(id,operation,input),
  inspect:()=>core.inspectLab(id)
 });
})(typeof window!=="undefined"?window:{});
