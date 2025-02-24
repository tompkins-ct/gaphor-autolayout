// const process = require('node:process');

const ELK = require('elkjs')
const elk = new ELK()

function elk_gen() {
  return new ELK()
}

const args = process.argv.slice(2);
let json = JSON.parse(args[0])

function process_layout(json) {
  console.log(JSON.stringify(json));
}

elk.layout(json).then(process_layout).catch(console.log);

// function  layout_json(json) {
//   const result = elk.layout(json)
//   return result.toString()
// }
//
// module.exports = {
//   elk_gen,
//   layout_json
// }
