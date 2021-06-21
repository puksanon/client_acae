const zlib = require("zlib");
const fs = require("fs");
const base64converter = require("./base64_converter");
const tf = require("@tensorflow/tfjs");
const tfq = require("@tensorflow/tfjs-node");

const inference = async (latent_vector) => {
  try {
    const model = await tf.loadLayersModel("https://raw.githubusercontent.com/Untesler/DCAE_Compressor/main/decoder_model/model.json");
    let tensor  =  tf.tensor(latent_vector).reshape([1, 2048]);
    let decoded = model.predict(tensor);
    decoded     =  decoded.mul(255).reshape([128, 128, 3]);
    return  tfq.node.encodePng(decoded)
  } catch (err) {
    console.error(err.message);
  }
};

const decode = async (compressed_bytes) => {
  try {
    let decompressed = await zlib.inflateSync(compressed_bytes);
    let decompressed_str = await decompressed.toString();
    let decom_latent = await decompressed_str.split(" ");
    decom_latent = await decom_latent.map((x) => parseFloat(x));
    decoded_image = await inference(decom_latent);
    base64_image = await base64converter.bufferToBase64(decoded_image);
    return await base64_image;
  } catch (err) {
    console.error(err);
  }
};

const converttoBuffer = (ab) => {
  var buf = Buffer.alloc(ab.byteLength);
  var view = new Uint8Array(ab);
  for (var i = 0; i < buf.length; ++i) {
      buf[i] = view[i];
  }
  return buf;
}

module.exports = { decode , converttoBuffer };