// const process = require('node:process');

const ELK = require('elkjs')
const elk = new ELK()

const args = process.argv.slice(2);
let json;

try {
  json = JSON.parse(args[0])
} catch (error) {
  console.error(`Failed to parse ELK input JSON: ${error.message}`);
  process.exit(1);
}

function process_layout(json) {
  console.log(JSON.stringify(json));
}

elk.layout(json).then(process_layout).catch(error => {
  console.error(`ELK layout failed: ${error.message}`);
  process.exit(1);
});

// function  layout_json(json) {
//   const result = elk.layout(json)
//   return result.toString()
// }
//
// module.exports = {
//   elk_gen,
//   layout_json
// }
