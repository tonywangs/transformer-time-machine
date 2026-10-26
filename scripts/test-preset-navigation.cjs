'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('dist/app.js','utf8');
const section = (a,b) => source.slice(source.indexOf(a),source.indexOf(b,source.indexOf(a)));
const base = JSON.parse(fs.readFileSync('dist/data/traces.json'));
base.scenes.sort((a,b)=>['story','echo','capital'].indexOf(a.id)-['story','echo','capital'].indexOf(b.id));
function harness(fetcher) {
  const elements=new Map();
  const $=id=>{if(!elements.has(id))elements.set(id,{value:'',replaceChildren(){},addEventListener(){},focus(){}});return elements.get(id);};
  const calls=[];
  const context=vm.createContext({$,Map,Promise,AbortController,setTimeout,clearTimeout,
    pause(){},initDataUI(){},render(){},renderCustomUI(){},validateData(){},
    scene(){return context.getState().data.scenes[context.getState().scene];},
    transport:{position:0},exampleOrder:['story','echo','capital'],
    fetch:fetcher || (async url=>({ok:true,json:async()=>({...structuredClone(base),model:{...base.model,id:`EleutherAI/pythia-${url.match(/examples-(.*)\.json/)[1]}-deduped`}})})),
    experiment:{cancel:async()=>{calls.push('cancel');},start:(...args)=>calls.push(args),
      onComplete:result=>{context.installCustom(result);}},
  });
  vm.runInContext(section('const state =','const motionPreference =')+
    section('function cancelAnalysis()',"$('sceneTabs').addEventListener('keydown'")+`
    modelCatalog = ${JSON.stringify(require('../dist/data/models.json'))};
    bundledData = ${JSON.stringify(base)}; state.data = bundledData; state.frame=20;
    presetModels.set('70m',bundledData);
    // Tiny navigation fixture: deliberately reuse steps; numerical validity is tested separately.
    for (const size of ['160m','410m','1b']) modelCatalog[size].checkpoints=modelCatalog['70m'].checkpoints;
    globalThis.getState=()=>state;
    globalThis.snapshot=()=>({tab:activeTab,custom:custom.selected,draft:$('customPromptInput').value,model:state.data.model.id,scene:scene().id,frame:state.frame,loading:modelLoading});
    globalThis.installCustom=(result)=>{state.data={model:result.model,checkpoints:result.checkpoints,scenes:[result]};state.scene=0;customResults.set(selectedModel,result);};
    globalThis.seedCustom=(size,result)=>customResults.set(size,result);
  `,context);
  return {context,$,calls};
}
test('all preset tabs remain presets for each model without overwriting a draft or starting inference',async()=>{
  const h=harness();h.$('customPromptInput').value='My unfinished sentence';
  for(const size of ['160m','410m','1b','70m']){
    await h.context.changeModel(size);
    for(const id of ['story','echo','capital']){
      h.context.selectTab(id);
      const s=h.context.snapshot();
      assert.equal(s.tab,id);assert.equal(s.scene,id);assert.equal(s.custom,false);
      assert.equal(s.draft,'My unfinished sentence');assert.equal(s.model,`EleutherAI/pythia-${size}-deduped`);
      assert.equal(s.frame,20);
    }
  }
  assert.equal(h.calls.filter(Array.isArray).length,0);
});
test('returning to Your prompt restores its model-specific result and preserves an edited draft',()=>{
  const h=harness(); const result={...base.scenes[0],id:'custom-test',custom:true,model:base.model,checkpoints:base.checkpoints};
  h.context.seedCustom('70m',result);h.$('customPromptInput').value='Edited draft';
  h.context.selectTab('capital');h.context.selectTab('custom');
  assert.equal(h.context.snapshot().scene,'custom-test');assert.equal(h.context.snapshot().draft,'Edited draft');
});
test('a late model download cannot replace the newer selected model',async()=>{
  const pending={};const h=harness(url=>new Promise(resolve=>{pending[url]=resolve;}));
  const older=h.context.changeModel('160m'); const newer=h.context.changeModel('410m');
  const reply=size=>({ok:true,json:async()=>({...structuredClone(base),model:{...base.model,id:`EleutherAI/pythia-${size}-deduped`}})});
  pending['data/examples-410m.json'](reply('410m'));await newer;
  pending['data/examples-160m.json'](reply('160m'));await older;
  assert.equal(h.context.snapshot().model,'EleutherAI/pythia-410m-deduped');
});
test('tab changes during a model download determine the final displayed preset',async()=>{
  let finish;const h=harness(()=>new Promise(resolve=>{finish=resolve;}));
  const loading=h.context.changeModel('160m');h.context.selectTab('capital');
  finish({ok:true,json:async()=>({...structuredClone(base),model:{...base.model,id:'EleutherAI/pythia-160m-deduped'}})});
  await loading;assert.equal(h.context.snapshot().scene,'capital');assert.equal(h.context.snapshot().custom,false);
});
test('leaving a pending custom start cannot start inference after cancellation settles',async()=>{
  const h=harness();const run=h.context.runSelectedModel('Hello');h.context.selectTab('story');await run;
  assert.equal(h.calls.filter(Array.isArray).length,0);assert.equal(h.context.snapshot().tab,'story');
});

test('failed preset loading preserves the displayed model and does not overwrite the draft',async()=>{
  const h=harness(async()=>{throw new Error('Offline');});h.$('customPromptInput').value='Keep this';
  h.context.selectTab('capital');await h.context.changeModel('160m');
  assert.equal(h.context.snapshot().model,base.model.id);assert.equal(h.context.snapshot().tab,'capital');
  assert.equal(h.context.snapshot().draft,'Keep this');assert.equal(h.context.snapshot().loading,false);
  await h.context.changeModel('70m');assert.equal(h.context.snapshot().scene,'capital');
});
