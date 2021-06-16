const zlib = require("zlib");
const fs = require("fs");
const base64converter = require("./base64_converter");
// const tf = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const inference = async (latent_vector) => {
  try {
    const model = await tf.loadLayersModel("https://raw.githubusercontent.com/Untesler/DCAE_Compressor/main/decoder_model/model.json");
    let tensor  =  tf.tensor(latent_vector).reshape([1, 2048]);
    let decoded = model.predict(tensor);
    decoded     =  decoded.mul(255).reshape([128, 128, 3]);
    // return  tf.node.encodePng(decoded);
    return decoded.dataSync()
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

const decodeImage = async (image) => {
  compressed_bytes = fs.readFileSync(image);
  console.log((fs.readFileSync(image)))
    try {
        compressed_bytes  = await fs.readFileSync(image);
        base64 = await decode(compressed_bytes);
        return base64
    } catch (error) {
        console.error(error);
        return error
    }
};

module.exports = { decodeImage , decode};