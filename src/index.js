const zlib = require("zlib");
const fs = require("fs");
const base64converter = require("./base64_converter");
const tf = require("@tensorflow/tfjs-node");

const inference = async (latent_vector) => {
  try {
    const model = await tf.loadLayersModel("file://decoder_model/model.json");
    let tensor = tf.tensor(latent_vector).reshape([1, 2048]);
    let decoded = model.predict(tensor);
    decoded = decoded.mul(255).reshape([128, 128, 3]);
    return tf.node.encodePng(decoded);
  } catch (err) {
    console.error(err.message);
  }
};

module.exports = async (compressed_bytes) => {
  try {
    let decompressed = zlib.inflateSync(compressed_bytes);
    let decompressed_str = decompressed.toString();
    let decom_latent = decompressed_str.split(" ");
    decom_latent = decom_latent.map((x) => parseFloat(x));
    decoded_image = await inference(decom_latent);
    base64_image = base64converter.bufferToBase64(decoded_image);
    return base64_image;
  } catch (err) {
    console.error(err);
  }
};

