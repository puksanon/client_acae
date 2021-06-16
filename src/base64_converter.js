const bufferToBase64 = (buff) => {
  let bin = Array.prototype.map
    .call(buff, (ch) => {
      return String.fromCharCode(ch);
    })
    .join("");
  return btoa(bin);
};

const base64ToBuffer = (base64) => {
  let bin = atob(base64);
  let buff = new Uint8Array(bin.length);
  Array.prototype.forEach.call(bin, (ch, i) => {
    buff[i] = ch.charCodeAt(0);
  });
  return buff;
};

module.exports = { bufferToBase64, base64ToBuffer };
