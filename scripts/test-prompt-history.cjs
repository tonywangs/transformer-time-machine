const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const source=fs.readFileSync('dist/app.js','utf8');
const restore=source.slice(source.indexOf('async function restorePrompt('),source.indexOf("$('savedPromptSelect').addEventListener"));
function harness(get) {
 const input={value:'new draft'}, installed=[];
 const context={requestIntent:0,selectedModel:'1b',state:{frame:20},custom:{},completedModels:new Map(),modelCatalog:{'1b':{model:{id:'EleutherAI/pythia-1b-deduped'}}},promptHistory:{get},$:()=>input,renderCustomUI(){},experiment:{onComplete:result=>installed.push(result)}};
 context.cancelAnalysis=()=>{context.requestIntent++};
 vm.createContext(context);vm.runInContext(restore,context);
 return {context,input,installed};
}
const result={prompt:'old prompt',model:{id:'EleutherAI/pythia-1b-deduped'}};
test('reopen a persisted result without inference and preserve comparison checkpoint',async()=>{
 const h=harness(async()=>result);await h.context.restorePrompt('1b:old prompt');
 assert.equal(h.input.value,'old prompt');assert.equal(h.context.custom.resumeFrame,20);assert.deepEqual(h.installed,[result]);
});
test('late history reads cannot replace a newer model selection',async()=>{
 let resolve;const h=harness(()=>new Promise(r=>resolve=r));const pending=h.context.restorePrompt('1b:old prompt');
 h.context.selectedModel='70m';h.context.cancelAnalysis();resolve(result);await pending;
 assert.equal(h.input.value,'new draft');assert.equal(h.installed.length,0);
});
test('unavailable saved record leaves the current draft intact',async()=>{
 const h=harness(async()=>undefined);await h.context.restorePrompt('missing');
 assert.equal(h.input.value,'new draft');assert.equal(h.installed.length,0);assert.equal(h.context.custom.statusKind,'error');
});
