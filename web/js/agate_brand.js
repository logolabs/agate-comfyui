/**
 * Agate branding: the Agate mark (the geometric "A") above the AGATE wordmark,
 * drawn in a reserved strip at the bottom of every Agate node. Agate Red on the node's
 * own ground, so it reads on light and dark ComfyUI themes alike. Purely visual -- remove
 * this file (and WEB_DIRECTORY in __init__.py) and the nodes work exactly as before.
 */
import { app } from "../../scripts/app.js";

const BRAND = {
  mark: "data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%22230%20185%20790%20725%22%20width%3D%22790%22%20height%3D%22725%22%3E%3Cpath%20d%3D%22M710.22%2C691.59L712.02%2C694.91L740.39%2C781.48C748.49%2C809.67%20758.31%2C827.63%20744.68%2C838.68L712.80%2C866.81L678.39%2C897.42C694.72%2C899.01%20838.35%2C899.19%201006.51%2C898.22L1006.24%2C896.91C987.39%2C877.21%20964.30%2C853.50%20958.92%2C847.09A90.44%2C90.44%200.000%200%2C1%20936.06%2C809.93L923.94%2C777.08C900.99%2C714.87%20811.30%2C460.36%20777.65%2C368.50L770.91%2C350.12L715.10%2C196.63L704.50%2C196.35L502.52%2C196.29C515.87%2C216.41%20527.86%2C229.10%20539.05%2C244.94A16.30%2C16.30%200.000%200%2C1%20539.67%2C260.66L458.09%2C459.12C440.16%2C502.57%20322.00%2C790.09%20313.43%2C809.47L307.28%2C823.48C303.25%2C831.67%20299.94%2C837.30%20280.58%2C856.58L239.04%2C897.33C298.30%2C900.29%20428.89%2C898.06%20476.36%2C898.26L476.76%2C897.28L414.27%2C844.73C398.95%2C831.84%20397.32%2C830.00%20407.99%2C799.04C415.73%2C777.98%20442.60%2C696.28%20445.77%2C691.55L605.50%2C691.44L710.22%2C691.59ZM584.84%2C332.26L576.70%2C352.58C557.97%2C399.52%20490.85%2C572.86%20479.88%2C601.87L466.26%2C637.19L491.85%2C637.60L692.53%2C637.19L686.27%2C618.61C668.49%2C566.43%20597.68%2C367.32%20589.63%2C345.52L584.84%2C332.26Z%22%20fill%3D%22%23F4291F%22%20fill-rule%3D%22evenodd%22%2F%3E%3C%2Fsvg%3E",
  wordmark: "AGATE",
  color: "#f4291f",
  strip: 52, // reserved at the bottom of the node, mark + gap + tracked caps
};

const mark = new Image();
let markReady = false;
mark.onload = () => {
  markReady = true;
};
mark.src = BRAND.mark;

app.registerExtension({
  name: "agate.brand",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name || !String(nodeData.name).startsWith("Agate")) return;

    // Reserve the strip in the size the frontend computes from the widgets, so the
    // brand never sits on top of one.
    const originalComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      const size = originalComputeSize
        ? originalComputeSize.apply(this, arguments)
        : out || [0, 0];
      return [size[0], size[1] + BRAND.strip];
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      if (!ctx || (this.flags && this.flags.collapsed)) return;
      const w = this.size && this.size[0];
      const h = this.size && this.size[1];
      if (!w || !h) return;
      const cx = w / 2;
      const markH = 28;
      const markW = markH * (790 / 725); // the mark's own viewBox aspect
      const top = h - BRAND.strip + 5;
      ctx.save();
      if (markReady) {
        ctx.drawImage(mark, cx - markW / 2, top, markW, markH);
      }
      ctx.fillStyle = BRAND.color;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      try {
        ctx.letterSpacing = "3px";
      } catch (e) {
        /* older canvases: the wordmark just reads tighter */
      }
      ctx.font = "600 10px ui-monospace, Consolas, Menlo, monospace";
      ctx.fillText(BRAND.wordmark, cx + 1.5, h - 7);
      try {
        ctx.letterSpacing = "0px";
      } catch (e) {
        /* as above */
      }
      ctx.restore();
    };
  },
});
