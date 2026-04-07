import Replicate from "replicate";

export default async function handler(req: any, res: any) {
  // Allow CORS if needed, though Vercel handles same-origin automatically
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { prompt, image, aspectRatio, resolution, negativePrompt, numImages = 1, model = "google/nano-banana-pro" } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }

    const replicateApiToken = process.env.VITE_REPLICATE_API_TOKEN;
    if (!replicateApiToken) {
      return res.status(500).json({ error: "VITE_REPLICATE_API_TOKEN is not set" });
    }

    const replicate = new Replicate({
      auth: replicateApiToken,
    });

    // Translate prompt to English using Gemini
    let finalPrompt = prompt;
    try {
      if (process.env.GEMINI_API_KEY) {
        const { GoogleGenAI } = await import("@google/genai");
        const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        const response = await ai.models.generateContent({
          model: "gemini-3-flash-preview",
          contents: `Translate the following text to English. If it is already in English, just return it as is. Only return the translated text, nothing else.\n\nText: ${prompt}`,
        });
        if (response.text) {
          finalPrompt = response.text.trim();
          console.log(`Original prompt: ${prompt}\nTranslated prompt: ${finalPrompt}`);
        }
      }
    } catch (translateError: any) {
      console.warn("Translation skipped or failed:", translateError.message || translateError);
      // Fallback to original prompt if translation fails
    }

    if (negativePrompt && negativePrompt.trim() !== "") {
      finalPrompt += `\nNegative prompt: ${negativePrompt}`;
    }

    const input: any = { 
      prompt: finalPrompt
    };

    if (model === "black-forest-labs/flux-2-pro") {
      input.safety_tolerance = 2;
      if (aspectRatio) {
        input.aspect_ratio = aspectRatio;
      }
      if (resolution) {
        if (resolution === "1K") input.resolution = "1 MP";
        else if (resolution === "2K") input.resolution = "2 MP";
        else if (resolution === "4K") input.resolution = "4 MP";
        else input.resolution = resolution;
      }
      if (image) {
        input.input_images = [image];
        if (!aspectRatio) {
          input.aspect_ratio = "match_input_image";
        }
      }
    } else {
      input.safety_filter_level = "block_only_high";
      input.allow_fallback_model = true;
      if (aspectRatio) {
        input.aspect_ratio = aspectRatio;
      }
      if (resolution) {
        input.resolution = resolution;
      }
      if (image) {
        input.image_input = [image];
      }
    }

    const count = Math.min(Math.max(Number(numImages) || 1, 1), 4);
    const results = [];
    
    // Run sequentially to avoid rate limit bursts
    for (let i = 0; i < count; i++) {
      try {
        const output = await replicate.run(model as `${string}/${string}`, { input });
        results.push(output);
      } catch (err: any) {
        if (err.status === 429 && i > 0) {
          console.warn("Rate limit hit, returning partial results");
          break;
        }
        throw err;
      }
    }

    const outputs = results.map((output: any) => {
      let imageUrl = output;
      if (output && typeof output.url === 'function') {
        imageUrl = output.url().toString();
      } else if (Array.isArray(output) && output.length > 0) {
        if (typeof output[0].url === 'function') {
          imageUrl = output[0].url().toString();
        } else {
          imageUrl = output[0];
        }
      }
      return imageUrl;
    });

    console.log("Replicate outputs:", outputs);

    res.status(200).json({ outputs });
  } catch (error: any) {
    console.error("Error generating image:", error);
    res.status(500).json({ error: error.message || "Failed to generate image" });
  }
}
