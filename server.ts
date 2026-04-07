import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import Replicate from "replicate";
import { GoogleGenAI } from "@google/genai";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // API routes FIRST
  app.post("/api/generate", async (req, res) => {
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
      let translatedPrompt = prompt;
      try {
        if (process.env.GEMINI_API_KEY) {
          const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
          const response = await ai.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: `Translate the following text to English. If it is already in English, just return it as is. Only return the translated text, nothing else.\n\nText: ${prompt}`,
          });
          if (response.text) {
            translatedPrompt = response.text.trim();
            console.log(`Original prompt: ${prompt}\nTranslated prompt: ${translatedPrompt}`);
          }
        }
      } catch (translateError: any) {
        console.warn("Translation skipped or failed:", translateError.message || translateError);
        // Fallback to original prompt if translation fails
      }

      let finalPrompt = translatedPrompt;
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
        // Default to nano-banana-pro
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

      res.json({ outputs });
    } catch (error: any) {
      console.error("Error generating image:", error);
      res.status(500).json({ error: error.message || "Failed to generate image" });
    }
  });

  app.post("/api/upscale", async (req, res) => {
    try {
      const { image, scale = 4, faceEnhance = false } = req.body;
      if (!image) {
        return res.status(400).json({ error: "Image is required for upscaling" });
      }

      const replicateApiToken = process.env.VITE_REPLICATE_API_TOKEN;
      if (!replicateApiToken) {
        return res.status(500).json({ error: "VITE_REPLICATE_API_TOKEN is not set" });
      }

      const replicate = new Replicate({
        auth: replicateApiToken,
      });

      const input = {
        image,
        scale: Number(scale),
        face_enhance: Boolean(faceEnhance)
      };

      const output: any = await replicate.run("nightmareai/real-esrgan", { input });
      
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

      res.json({ output: imageUrl });
    } catch (error: any) {
      console.error("Error upscaling image:", error);
      res.status(500).json({ error: error.message || "Failed to upscale image" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
