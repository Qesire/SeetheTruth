'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const Audit=require('../universal_surface_audit.js'),root=path.resolve(__dirname,'..');
const src=Object.fromEntries(Audit.screens.map(row=>[row.page,fs.readFileSync(path.join(root,row.page),'utf8')]));
const evidence=Audit.analyze(src);
assert.equal(evidence.rows.length,20);
assert.equal(evidence.counts.tensor,2);assert.equal(evidence.counts.state,7);
assert.ok(evidence.counts.specialized>=1,'at least one legacy specialized workflow must remain explicitly classified until migrated');
assert.equal(evidence.counts.total,evidence.counts.tensor+evidence.counts.state+evidence.counts.specialized+evidence.counts.navigation+evidence.counts.desktopOnly,'every screen must be accounted for');
assert.equal(new Set(Audit.screens.map(x=>x.page)).size,evidence.rows.length);
for(const row of evidence.rows){
 assert.ok(row.note.length>=15,row.page+' missing explicit limitation');
 if(['tensor','state'].includes(row.status)){assert.equal(row.actual,row.status);}
 else assert.equal(row.actual,'none');
}
const allow=new Set(fs.readFileSync(path.join(root,'android/teaching-assets.txt'),'utf8').trim().split(/\r?\n/));
for(const row of evidence.rows)if(row.status!=='desktop_only'){
 assert.ok(allow.has(row.page),'missing Android teaching page: '+row.page);
}
for(const file of ['state_microscope.js','state_microscope_ui.js','state_microscope.css','universal_surface_audit.js'])assert.ok(allow.has(file),file);
const cap=src['capabilities.html'];assert.ok(cap.includes('id="surfaceAuditRows"'));
assert.ok(cap.includes('universal_surface_audit.js'));
const accidental={...src,'learn.html':src['learn.html'].replace('<script src="compute_microscope.js"></script>','')};
assert.throws(()=>Audit.analyze(accidental),/universal capability regression/);
const stale={...src,'paper_lab.html':src['paper_lab.html'].replace('<script src="paper_patches.js"></script>','<script src="compute_microscope.js"></script><script src="compute_microscope_ui.js"></script><script src="paper_patches.js"></script>')};
assert.throws(()=>Audit.analyze(stale),/audit status stale/);
const missing={...src};delete missing['paper_lab.html'];assert.throws(()=>Audit.analyze(missing),/missing actual page/);
console.log('SURFACE_AUDIT_PASS '+JSON.stringify(evidence.counts)+' core adapters tested; independent surfaces stay clearly flagged');
