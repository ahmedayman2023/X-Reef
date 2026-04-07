import { GoogleGenAI } from "@google/genai";
async function test() {
  try {
    const ai = new GoogleGenAI({ apiKey: undefined });
    await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: "Hello",
    });
    console.log("Success");
  } catch (e: any) {
    console.error("Error:", e.message);
  }
}
test();
