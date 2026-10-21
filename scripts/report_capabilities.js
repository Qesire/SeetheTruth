'use strict';
const A=require('../block_capabilities.js').audit();
if(process.argv.includes('--json'))console.log(JSON.stringify(A,null,2));
else {console.log('SeetheTruth / '+A.schema);for(const [k,v] of Object.entries(A.counts))console.log(k.padEnd(22),v);
 console.log('NumPy missing: '+A.rows.filter(r=>!r.numpyEligible).map(r=>r.kind).join(', '));
 console.log('Torch missing: '+A.rows.filter(r=>!r.torchEligible).map(r=>r.kind).join(', '));
 console.log('Kind eligibility is not a graph/gradient/model equivalence proof.');
}
