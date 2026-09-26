// Run with node audit/test_range_controls.cjs. No browser or external packages needed.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'',textContent:'',disabled:false,
    showModal(){this.open=true;},close(){this.open=false;}});
  return elements.get(id);
}
const context = vm.createContext({document:{getElementById:element,addEventListener(){}},
  location:{search:'?token=test'},URLSearchParams,clearTimeout(){},
  fetch:()=>new Promise(()=>{})});
const script = fs.readFileSync(__dirname+'/review_gt_artifacts.html','utf8').split('<script>')[1].split('</script>')[0];
vm.runInContext(script, context);
vm.runInContext('state={frames:[{video_frame:2436},{video_frame:2437}]};saving=true;',context);
element('rangeStart').value='2436';element('rangeEnd').value='2437';element('rangeBird').value='14';
element('rangeFlag').onclick();
assert.equal(element('rangeConfirm').open,true,'A pending frame save must not swallow the click');
assert.match(element('rangeQuestion').textContent,/2436–2437 inclusive \(2 frames\)/);
assert.match(element('rangeQuestion').textContent,/GT issue reported/,'Blank optional reason must not block');
element('rangeYes').onclick();
assert.match(element('rangeSaveStatus').textContent,/still finishing/);
element('rangeStart').value='2438';element('rangeFlag').onclick();
assert.match(element('rangeStatus').textContent,/start no greater than end/);
console.log('Range-control regression checks passed');
