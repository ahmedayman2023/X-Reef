import React, { useState, useRef, useEffect } from "react";
import { Loader2, Image as ImageIcon, Upload, X, Download, Sparkles, ChevronDown, ChevronUp, Maximize2, Clock, Trash2, Crop, Zap, ImagePlus, Library, Edit2, Plus, Save, LayoutTemplate, Settings2 } from "lucide-react";
import ReactCrop, { type Crop as CropType } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

interface HistoryItem {
  id: string;
  url: string;
  prompt: string;
  timestamp: number;
}

const DEFAULT_PROMPT_BANK = [
  {
    category: "خيال علمي (Sci-Fi)",
    prompts: [
      { title: "مدينة مستقبلية", prompt: "A futuristic cyberpunk city at night, neon lights, flying cars, highly detailed, 8k resolution, cinematic lighting, Unreal Engine 5 render" },
      { title: "رائد فضاء", prompt: "A highly detailed portrait of an astronaut in a futuristic glowing suit, exploring an alien planet with bioluminescent flora, cinematic, hyperrealistic" },
      { title: "محطة فضائية", prompt: "A massive space station orbiting a gas giant, highly detailed, cinematic lighting, sci-fi concept art, 8k" }
    ]
  },
  {
    category: "طبيعة ومناظر (Nature)",
    prompts: [
      { title: "غابة سحرية", prompt: "A magical enchanted forest with glowing mushrooms, ancient giant trees, ethereal mist, sun rays filtering through leaves, fantasy concept art, trending on ArtStation" },
      { title: "جبال جليدية", prompt: "Majestic snow-capped mountains at sunset, a crystal clear lake reflecting the sky, aurora borealis, photorealistic, National Geographic photography" },
      { title: "شاطئ استوائي", prompt: "A beautiful tropical beach at golden hour, crystal clear turquoise water, palm trees, white sand, highly detailed, photorealistic" }
    ]
  },
  {
    category: "شخصيات (Characters)",
    prompts: [
      { title: "محارب ساموراي", prompt: "A fierce cyberpunk samurai warrior with a glowing katana, wearing high-tech armor, standing in the rain, dramatic lighting, highly detailed character design" },
      { title: "أميرة خيالية", prompt: "A beautiful elven princess with silver hair, wearing an intricate elegant gown, standing in a magical garden, soft lighting, ethereal, fantasy portrait" },
      { title: "ساحر غامض", prompt: "A mysterious wizard casting a glowing spell, dark robes, glowing eyes, magical particles, highly detailed, fantasy concept art" }
    ]
  },
  {
    category: "أنمي (Anime)",
    prompts: [
      { title: "مشهد كلاسيكي", prompt: "Studio Ghibli style anime scenery, a cozy small bakery in a quiet Japanese town, warm sunlight, highly detailed, beautiful colors, nostalgic feeling" },
      { title: "شخصية قتالية", prompt: "Epic anime fight scene, a powerful hero with glowing aura, dynamic pose, shattered ground, intense action, colorful energy blasts, 4k" },
      { title: "فتاة الأنمي", prompt: "A beautiful anime girl with long flowing hair, wearing a school uniform, standing under a cherry blossom tree, petals falling, soft lighting, highly detailed" }
    ]
  },
  {
    category: "تصميم داخلي (Interior)",
    prompts: [
      { title: "غرفة معيشة حديثة", prompt: "A modern minimalist living room with large windows, natural light, cozy furniture, indoor plants, architectural photography, highly detailed" },
      { title: "مقهى دافئ", prompt: "A cozy rustic coffee shop interior, warm lighting, wooden furniture, espresso machine, people reading, highly detailed, photorealistic" }
    ]
  }
];

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [imageFile, setImageFile] = useState<string | null>(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Advanced Settings State
  const [negativePrompt, setNegativePrompt] = useState("");
  const [aspectRatio, setAspectRatio] = useState("1:1");
  const [resolution, setResolution] = useState("1K");
  const [numImages, setNumImages] = useState(1);
  const [model, setModel] = useState("google/nano-banana-pro");
  
  // Fullscreen State
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // Cropping State
  const [imageToCrop, setImageToCrop] = useState<string | null>(null);
  const [crop, setCrop] = useState<CropType>({ unit: '%', x: 10, y: 10, width: 80, height: 80 });
  const [completedCrop, setCompletedCrop] = useState<any>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // Loading Animation State
  const [progress, setProgress] = useState(0);
  const [loadingText, setLoadingText] = useState("تحليل الوصف...");

  // Upscale State
  const [isUpscaling, setIsUpscaling] = useState<string | null>(null);

  // Prompt Bank State
  const [isPromptBankOpen, setIsPromptBankOpen] = useState(false);
  const [promptBank, setPromptBank] = useState(() => {
    const saved = localStorage.getItem('xreef_prompt_bank');
    return saved ? JSON.parse(saved) : DEFAULT_PROMPT_BANK;
  });
  const [isPromptBankEditMode, setIsPromptBankEditMode] = useState(false);
  const [editingPrompt, setEditingPrompt] = useState<{catIdx: number, promptIdx: number, title: string, prompt: string} | null>(null);
  const [isAddingCategory, setIsAddingCategory] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState("");

  // Template State
  const [isTemplateModalOpen, setIsTemplateModalOpen] = useState(false);
  const [templateImage, setTemplateImage] = useState<string | null>(() => {
    try { return localStorage.getItem('xreef_template_image'); } catch { return null; }
  });
  const [templateSettings, setTemplateSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('xreef_template_settings');
      return saved ? JSON.parse(saved) : { scale: 70, offsetX: 55, offsetY: 50 };
    } catch {
      return { scale: 70, offsetX: 55, offsetY: 50 };
    }
  });
  const [imageToTemplate, setImageToTemplate] = useState<string | null>(null);
  const templateCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    localStorage.setItem('xreef_prompt_bank', JSON.stringify(promptBank));
  }, [promptBank]);

  useEffect(() => {
    localStorage.setItem('xreef_template_settings', JSON.stringify(templateSettings));
  }, [templateSettings]);

  useEffect(() => {
    if (!isTemplateModalOpen || !templateImage || !imageToTemplate || !templateCanvasRef.current) return;

    const canvas = templateCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const tImg = new Image();
    tImg.crossOrigin = "anonymous";
    tImg.onload = () => {
      canvas.width = tImg.width;
      canvas.height = tImg.height;
      
      // Draw template
      ctx.drawImage(tImg, 0, 0);

      const gImg = new Image();
      gImg.crossOrigin = "anonymous";
      gImg.onload = () => {
        const targetWidth = (tImg.width * templateSettings.scale) / 100;
        const ratio = gImg.height / gImg.width;
        const targetHeight = targetWidth * ratio;

        const x = (tImg.width * templateSettings.offsetX) / 100 - (targetWidth / 2);
        const y = (tImg.height * templateSettings.offsetY) / 100 - (targetHeight / 2);

        ctx.drawImage(gImg, x, y, targetWidth, targetHeight);
      };
      gImg.src = imageToTemplate;
    };
    tImg.src = templateImage;

  }, [templateImage, imageToTemplate, templateSettings, isTemplateModalOpen]);

  const handleTemplateUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const base64 = event.target?.result as string;
      setTemplateImage(base64);
      try {
        localStorage.setItem('xreef_template_image', base64);
      } catch (err) {
        console.warn("Template image too large for localStorage");
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDownloadTemplate = () => {
    if (!templateCanvasRef.current) return;
    const dataUrl = templateCanvasRef.current.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `xreef-template-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const openTemplateModal = (imageUrl: string) => {
    setImageToTemplate(imageUrl);
    setIsTemplateModalOpen(true);
  };

  const handleSavePrompt = () => {
    if (!editingPrompt || !editingPrompt.title.trim() || !editingPrompt.prompt.trim()) return;
    const newBank = [...promptBank];
    if (editingPrompt.promptIdx === -1) {
      newBank[editingPrompt.catIdx].prompts.push({ title: editingPrompt.title, prompt: editingPrompt.prompt });
    } else {
      newBank[editingPrompt.catIdx].prompts[editingPrompt.promptIdx] = { title: editingPrompt.title, prompt: editingPrompt.prompt };
    }
    setPromptBank(newBank);
    setEditingPrompt(null);
  };

  const handleDeletePrompt = (catIdx: number, promptIdx: number) => {
    if(!confirm("هل أنت متأكد من حذف هذا الوصف؟")) return;
    const newBank = [...promptBank];
    newBank[catIdx].prompts.splice(promptIdx, 1);
    setPromptBank(newBank);
  };

  const handleAddCategory = () => {
    if (!newCategoryName.trim()) return;
    setPromptBank([...promptBank, { category: newCategoryName, prompts: [] }]);
    setNewCategoryName("");
    setIsAddingCategory(false);
  };

  const handleDeleteCategory = (catIdx: number) => {
    if(!confirm("هل أنت متأكد من حذف هذا التصنيف بالكامل؟")) return;
    const newBank = [...promptBank];
    newBank.splice(catIdx, 1);
    setPromptBank(newBank);
  };

  useEffect(() => {
    if (!isLoading) {
      setProgress(0);
      return;
    }

    const duration = 12000; // 12 seconds simulated
    const interval = 100;
    const steps = duration / interval;
    let currentStep = 0;

    const timer = setInterval(() => {
      currentStep++;
      // Easing function for progress (starts fast, slows down)
      const rawProgress = currentStep / steps;
      const easedProgress = 1 - Math.pow(1 - rawProgress, 3); // Cubic ease out
      
      const currentProgress = Math.min(99, Math.floor(easedProgress * 100));
      setProgress(currentProgress);

      if (currentProgress < 25) setLoadingText("تحليل الوصف وبناء المشهد...");
      else if (currentProgress < 50) setLoadingText("توليد التكوين الأساسي...");
      else if (currentProgress < 75) setLoadingText("إضافة التفاصيل والألوان...");
      else setLoadingText("اللمسات الأخيرة وتحسين الجودة...");

    }, interval);

    return () => clearInterval(timer);
  }, [isLoading]);

  // History State
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem('xreef_history');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem('xreef_history', JSON.stringify(history));
  }, [history]);

  const addToHistory = (urls: string[], currentPrompt: string) => {
    const newItems = urls.map(url => ({
      id: Math.random().toString(36).substring(2, 9),
      url,
      prompt: currentPrompt,
      timestamp: Date.now()
    }));
    setHistory(prev => [...newItems, ...prev].slice(0, 50)); // Keep last 50 images
  };

  const compressImage = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target?.result as string;
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const MAX_WIDTH = 1024;
          const MAX_HEIGHT = 1024;
          let width = img.width;
          let height = img.height;

          if (width > height) {
            if (width > MAX_WIDTH) {
              height *= MAX_WIDTH / width;
              width = MAX_WIDTH;
            }
          } else {
            if (height > MAX_HEIGHT) {
              width *= MAX_HEIGHT / height;
              height = MAX_HEIGHT;
            }
          }

          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            reject(new Error("Could not get canvas context"));
            return;
          }
          ctx.drawImage(img, 0, 0, width, height);
          
          // Export as JPEG with 80% quality to drastically reduce size
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          resolve(dataUrl);
        };
        img.onerror = (error) => reject(error);
      };
      reader.onerror = (error) => reject(error);
    });
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        // Compress the image before saving it to state
        const compressedBase64 = await compressImage(file);
        setImageFile(compressedBase64);
        setError(null);
      } catch (err) {
        console.error("Error compressing image:", err);
        setError("حدث خطأ أثناء معالجة الصورة.");
      }
    }
  };

  const removeImage = () => {
    setImageFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    setImageUrls([]);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          prompt, 
          image: imageFile,
          aspectRatio,
          resolution,
          negativePrompt,
          numImages,
          model
        }),
      });

      let data;
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        console.error("Non-JSON response:", text);
        throw new Error(`الخادم لم يرجع استجابة صحيحة (ربما مسار API غير موجود). رمز الخطأ: ${response.status}`);
      }

      if (!response.ok) {
        throw new Error(data.error || "حدث خطأ أثناء توليد الصورة");
      }

      if (data.outputs && Array.isArray(data.outputs)) {
        setImageUrls(data.outputs);
        addToHistory(data.outputs, prompt);
      } else if (typeof data.output === "string") {
        setImageUrls([data.output]);
        addToHistory([data.output], prompt);
      } else if (Array.isArray(data.output) && data.output.length > 0) {
        setImageUrls([data.output[0]]);
        addToHistory([data.output[0]], prompt);
      } else if (data.output && typeof data.output === "object" && data.output.url) {
        setImageUrls([data.output.url]);
        addToHistory([data.output.url], prompt);
      } else {
        console.error("Unexpected output format:", data.output);
        throw new Error(`تنسيق الاستجابة غير متوقع: ${JSON.stringify(data.output)}`);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "حدث خطأ أثناء توليد الصورة");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async (urlToDownload: string) => {
    if (!urlToDownload) return;
    try {
      // Fetch the image as a blob to force download instead of opening in new tab
      const response = await fetch(urlToDownload);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `XREEF-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Error downloading image:", err);
      // Fallback: open in new tab
      window.open(urlToDownload, '_blank');
    }
  };

  const handleUpscale = async (urlToUpscale: string) => {
    if (!urlToUpscale) return;
    setIsUpscaling(urlToUpscale);
    try {
      const response = await fetch("/api/upscale", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          image: urlToUpscale,
          scale: 4,
          faceEnhance: false
        }),
      });

      let data;
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        data = await response.json();
      } else {
        throw new Error(`الخادم لم يرجع استجابة صحيحة. رمز الخطأ: ${response.status}`);
      }

      if (!response.ok) {
        throw new Error(data.error || "حدث خطأ أثناء تكبير الصورة");
      }

      if (data.output) {
        // Replace the image in the current results if it's there
        setImageUrls(prev => prev.map(url => url === urlToUpscale ? data.output : url));
        
        // Replace in history
        setHistory(prev => prev.map(item => 
          item.url === urlToUpscale ? { ...item, url: data.output } : item
        ));

        // If it's the currently selected image, update it
        if (selectedImage === urlToUpscale) {
          setSelectedImage(data.output);
        }
      }
    } catch (err: any) {
      console.error("Error upscaling image:", err);
      alert(err.message || "حدث خطأ أثناء تكبير الصورة");
    } finally {
      setIsUpscaling(null);
    }
  };

  const handleUseAsInput = (url: string) => {
    setImageFile(url);
    setSelectedImage(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const openCropModal = async (url: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      setImageToCrop(blobUrl);
      setCrop({ unit: '%', x: 10, y: 10, width: 80, height: 80 });
      setCompletedCrop(null);
    } catch (err) {
      console.error("Error loading image for crop:", err);
      setImageToCrop(url);
    }
  };

  const closeCropModal = () => {
    if (imageToCrop && imageToCrop.startsWith('blob:')) {
      window.URL.revokeObjectURL(imageToCrop);
    }
    setImageToCrop(null);
  };

  const handleSaveCrop = async () => {
    if (!completedCrop || !imgRef.current || completedCrop.width === 0 || completedCrop.height === 0) {
      return;
    }

    const image = imgRef.current;
    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    
    canvas.width = completedCrop.width * scaleX;
    canvas.height = completedCrop.height * scaleY;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      completedCrop.width * scaleX,
      completedCrop.height * scaleY,
      0,
      0,
      completedCrop.width * scaleX,
      completedCrop.height * scaleY
    );

    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `XREEF-Cropped-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      closeCropModal();
    }, 'image/png');
  };

  return (
    <div 
      className="min-h-screen bg-cover bg-center bg-fixed relative flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8 font-sans" 
      dir="rtl"
      style={{ backgroundImage: "url('https://images.unsplash.com/photo-1774308667027-3ce5c579a518?q=80&w=2064&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')" }}
    >
      <style>{`
        @keyframes scan {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .animate-scan {
          animation: scan 2.5s linear infinite;
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 0.5; }
          100% { transform: scale(1.3); opacity: 0; }
        }
        .animate-pulse-ring {
          animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }
        @keyframes shimmer {
          0% { background-position: 20px 0; }
          100% { background-position: 0 0; }
        }
      `}</style>

      {/* Fullscreen Image Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-md p-4 sm:p-8"
          onClick={() => setSelectedImage(null)}
        >
          <button 
            className="absolute top-6 right-6 text-white/70 hover:text-white bg-white/10 hover:bg-white/20 rounded-full p-3 transition-all"
            onClick={() => setSelectedImage(null)}
          >
            <X className="w-6 h-6" />
          </button>
          
          <img 
            src={selectedImage} 
            alt="Fullscreen generated result" 
            className="max-w-full max-h-full object-contain rounded-xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
          
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 flex-wrap justify-center w-full px-4" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => handleUseAsInput(selectedImage)}
              className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-4 rounded-full font-bold transition-all shadow-lg hover:scale-105 border border-emerald-500"
            >
              <ImagePlus className="w-5 h-5" />
              استخدام كمرجع
            </button>
            <button
              onClick={() => handleUpscale(selectedImage)}
              disabled={isUpscaling === selectedImage}
              className="flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white px-6 py-4 rounded-full font-bold transition-all shadow-lg hover:scale-105 border border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUpscaling === selectedImage ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Zap className="w-5 h-5" />
              )}
              {isUpscaling === selectedImage ? "جاري التكبير..." : "تكبير الدقة"}
            </button>
            <button
              onClick={() => {
                setSelectedImage(null);
                openCropModal(selectedImage);
              }}
              className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-6 py-4 rounded-full font-bold transition-all shadow-lg hover:scale-105 border border-gray-600"
            >
              <Crop className="w-5 h-5" />
              قص
            </button>
            <button
              onClick={() => handleDownload(selectedImage)}
              className="flex items-center gap-3 bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-full font-bold transition-all shadow-[0_0_30px_rgba(37,99,235,0.5)] hover:scale-105"
            >
              <Download className="w-5 h-5" />
              تنزيل الصورة
            </button>
          </div>
        </div>
      )}

      {/* Cropping Modal */}
      {imageToCrop && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/95 backdrop-blur-md p-4 sm:p-8">
          <button 
            className="absolute top-6 right-6 text-white/70 hover:text-white bg-white/10 hover:bg-white/20 rounded-full p-3 transition-all z-10"
            onClick={closeCropModal}
          >
            <X className="w-6 h-6" />
          </button>
          
          <div className="flex flex-col items-center justify-center max-h-full max-w-full w-full">
            <h3 className="text-white text-xl font-bold mb-4 drop-shadow-md">قص الصورة</h3>
            <div className="overflow-auto max-h-[70vh] max-w-full bg-black/50 rounded-xl border border-blue-500/30 p-2 shadow-2xl">
              <ReactCrop 
                crop={crop} 
                onChange={(_, percentCrop) => setCrop(percentCrop)}
                onComplete={(c) => setCompletedCrop(c)}
                className="max-h-[65vh]"
              >
                <img 
                  ref={imgRef}
                  src={imageToCrop} 
                  alt="Crop me" 
                  className="max-h-[65vh] w-auto object-contain"
                />
              </ReactCrop>
            </div>
            
            <div className="mt-8 flex gap-4">
              <button
                onClick={closeCropModal}
                className="px-8 py-3 rounded-full font-bold text-white bg-gray-700 hover:bg-gray-600 transition-all border border-gray-500"
              >
                إلغاء
              </button>
              <button
                onClick={handleSaveCrop}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-full font-bold transition-all shadow-[0_0_30px_rgba(37,99,235,0.5)] hover:scale-105"
              >
                <Download className="w-5 h-5" />
                حفظ وتنزيل
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dark Overlay for better readability - Black theme */}
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm"></div>

      {/* Main Content */}
      <div className="relative z-10 max-w-6xl w-full space-y-10">
        
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-600 tracking-tight flex items-center justify-center gap-4 drop-shadow-lg">
            <Sparkles className="w-12 h-12 text-blue-500" />
            Xreef Version 1.0
          </h1>
          <p className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto font-light">
            أطلق العنان لخيالك. قم بتوليد وتعديل الصور باستخدام أحدث تقنيات الذكاء الاصطناعي.
          </p>
        </div>

        {/* Glassmorphism Card */}
        <div className="bg-black/40 backdrop-blur-xl border border-blue-500/20 p-6 md:p-10 rounded-[2.5rem] shadow-2xl grid grid-cols-1 lg:grid-cols-2 gap-10">
          
          {/* Left Column: Controls */}
          <div className="space-y-8 flex flex-col">
            <form onSubmit={handleGenerate} className="flex flex-col flex-grow space-y-6">
              
              {/* Prompt Input */}
              <div className="space-y-3">
                <div className="flex items-center justify-between ml-1">
                  <label htmlFor="prompt" className="block text-sm font-medium text-gray-200">
                    ماذا تريد أن ترسم اليوم؟
                  </label>
                  <button
                    type="button"
                    onClick={() => setIsPromptBankOpen(true)}
                    className="flex items-center gap-1.5 text-xs font-medium text-blue-400 hover:text-blue-300 transition-colors bg-blue-500/10 hover:bg-blue-500/20 px-3 py-1.5 rounded-full border border-blue-500/20"
                  >
                    <Library className="w-3.5 h-3.5" />
                    بنك الأوصاف
                  </button>
                </div>
                <textarea
                  id="prompt"
                  name="prompt"
                  rows={4}
                  className="appearance-none rounded-3xl relative block w-full px-6 py-4 bg-black/60 border border-blue-500/20 placeholder-gray-500 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-base resize-none transition-all shadow-inner"
                  placeholder="مثال: مدينة مستقبلية مضيئة بالنيون تحت المطر، ألوان سينمائية..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              {/* Advanced Settings (Always Visible) */}
              <div className="space-y-5 p-5 bg-black/40 border border-blue-500/20 rounded-3xl shadow-inner">
                {/* Negative Prompt */}
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-gray-400">
                    استبعاد من الصورة (Negative Prompt)
                  </label>
                  <input
                    type="text"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="مثال: نص، تشوه، ألوان باهتة..."
                    className="w-full px-4 py-2.5 bg-black/60 border border-blue-500/20 rounded-xl text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Model Selection */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-gray-400">
                      النموذج (Model)
                    </label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full px-4 py-2.5 bg-black/60 border border-blue-500/20 rounded-xl text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 appearance-none"
                    >
                      <option value="google/nano-banana-pro">Nano Banana Pro</option>
                      <option value="black-forest-labs/flux-2-pro">Flux 2 Pro</option>
                    </select>
                  </div>

                  {/* Aspect Ratio */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-gray-400">
                      أبعاد الصورة
                    </label>
                    <select
                      value={aspectRatio}
                      onChange={(e) => setAspectRatio(e.target.value)}
                      className="w-full px-4 py-2.5 bg-black/60 border border-blue-500/20 rounded-xl text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 appearance-none"
                    >
                      <option value="1:1">مربع (1:1)</option>
                      <option value="16:9">شاشة عريضة (16:9)</option>
                      <option value="9:16">طولي (9:16)</option>
                      <option value="4:3">أفقي (4:3)</option>
                      <option value="3:4">عمودي (3:4)</option>
                    </select>
                  </div>

                  {/* Resolution */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-gray-400">
                      الدقة (Resolution)
                    </label>
                    <select
                      value={resolution}
                      onChange={(e) => setResolution(e.target.value)}
                      className="w-full px-4 py-2.5 bg-black/60 border border-blue-500/20 rounded-xl text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 appearance-none"
                    >
                      <option value="1K">عادي (1K)</option>
                      <option value="2K">عالي (2K)</option>
                      <option value="4K">فائق (4K)</option>
                    </select>
                  </div>

                  {/* Num Images */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-gray-400">
                      عدد الصور
                    </label>
                    <select
                      value={numImages}
                      onChange={(e) => setNumImages(Number(e.target.value))}
                      className="w-full px-4 py-2.5 bg-black/60 border border-blue-500/20 rounded-xl text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 appearance-none"
                    >
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                      <option value={3}>3</option>
                      <option value={4}>4</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Image Upload */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-200 ml-1">
                  صورة مرجعية (اختياري)
                </label>
                {imageFile ? (
                  <div className="relative inline-block group">
                    <img src={imageFile} alt="Uploaded preview" className="h-32 w-32 object-cover rounded-2xl border border-blue-500/30 shadow-xl" />
                    <button
                      type="button"
                      onClick={removeImage}
                      className="absolute -top-3 -right-3 bg-red-500 text-white rounded-full p-1.5 hover:bg-red-600 transition-transform hover:scale-110 shadow-lg"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="flex justify-center px-6 pt-5 pb-6 border-2 border-blue-500/20 border-dashed rounded-3xl hover:border-blue-500 hover:bg-blue-900/10 cursor-pointer transition-all group bg-black/40"
                  >
                    <div className="space-y-2 text-center">
                      <div className="w-12 h-12 mx-auto bg-blue-500/10 rounded-full flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                        <Upload className="h-6 w-6 text-blue-500/50 group-hover:text-blue-400 transition-colors" />
                      </div>
                      <div className="flex text-sm text-gray-300 justify-center">
                        <span className="relative cursor-pointer font-medium text-blue-400 hover:text-blue-300">
                          اضغط لرفع صورة من جهازك
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept="image/*"
                  className="hidden"
                />
              </div>

              {/* Submit Button */}
              <div className="pt-2 mt-auto">
                <button
                  type="submit"
                  disabled={isLoading || !prompt.trim()}
                  className={`group relative w-full flex justify-center py-4 px-4 border border-transparent text-lg font-bold rounded-3xl text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 focus:ring-offset-black disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 ${
                    isLoading 
                      ? 'bg-blue-900/80 shadow-[0_0_30px_-5px_rgba(37,99,235,0.4)]' 
                      : 'bg-gradient-to-r from-blue-600 to-blue-800 hover:from-blue-500 hover:to-blue-700 shadow-[0_0_40px_-10px_rgba(37,99,235,0.5)] hover:shadow-[0_0_60px_-15px_rgba(37,99,235,0.7)]'
                  }`}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-3">
                      <div className="relative flex items-center justify-center w-6 h-6">
                        <div className="absolute w-full h-full border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                      </div>
                      جاري الإبداع...
                    </span>
                  ) : (
                    <span className="flex items-center gap-3">
                      <Sparkles className="w-6 h-6" />
                      توليد الصورة
                    </span>
                  )}
                </button>
              </div>
            </form>

            {/* Error Message */}
            {error && (
              <div className="rounded-2xl bg-red-500/10 p-5 border border-red-500/20 backdrop-blur-md">
                <div className="flex items-start">
                  <div className="ml-3">
                    <h3 className="text-sm font-bold text-red-400">حدث خطأ</h3>
                    <div className="mt-2 text-sm text-red-300/90 leading-relaxed">
                      <p>{error}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Result Area */}
          <div className="flex flex-col h-full">
            <div className="flex-grow rounded-[2rem] border border-blue-500/20 bg-black/60 overflow-hidden min-h-[400px] lg:min-h-[600px] flex items-center justify-center relative shadow-inner group">
              
              {/* Loading Overlay */}
              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-xl z-30 overflow-hidden transition-all duration-500">
                  {/* Diffusion Noise Simulation */}
                  <div 
                    className="absolute inset-0 mix-blend-overlay transition-all duration-200"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
                      transform: `scale(${1 + (progress / 100) * 0.5})`,
                      opacity: Math.max(0, 0.4 - (progress / 100) * 0.4)
                    }}
                  ></div>

                  {/* Background Glow */}
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(37,99,235,0.15)_0%,transparent_60%)] animate-pulse"></div>
                  
                  {/* Scanning Laser */}
                  <div className="absolute left-0 right-0 h-[2px] bg-blue-400 shadow-[0_0_20px_5px_rgba(59,130,246,0.6)] animate-scan z-0"></div>

                  <div className="relative flex items-center justify-center mb-8 z-10">
                    <div className="absolute w-24 h-24 border-t-2 border-r-2 border-blue-500 rounded-full animate-spin"></div>
                    <div className="absolute w-16 h-16 border-b-2 border-l-2 border-blue-300 rounded-full animate-[spin_1.5s_linear_infinite_reverse]"></div>
                    <div className="absolute w-24 h-24 border-2 border-blue-500/20 rounded-full animate-pulse-ring"></div>
                    <Sparkles className="w-8 h-8 text-blue-400 animate-pulse" />
                  </div>

                  <div className="relative z-10 flex flex-col items-center gap-3 w-full max-w-xs sm:max-w-sm px-4">
                    <h3 className="text-lg md:text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-200 animate-pulse tracking-wide text-center">
                      {loadingText}
                    </h3>
                    
                    {/* Progress Bar */}
                    <div className="w-full h-2.5 bg-gray-900/80 rounded-full mt-4 overflow-hidden border border-blue-500/30 shadow-inner">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300 ease-out relative"
                        style={{ width: `${progress}%` }}
                      >
                        <div className="absolute top-0 right-0 bottom-0 left-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.2)_50%,transparent_75%,transparent_100%)] bg-[length:10px_10px] animate-[shimmer_1s_linear_infinite]"></div>
                      </div>
                    </div>
                    <div className="flex justify-between w-full text-xs text-blue-300/70 mt-1 font-mono">
                      <span>{progress}%</span>
                      <span className="flex items-center gap-1">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        جاري المعالجة
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Image Display */}
              {imageUrls.length > 0 ? (
                <div className={`w-full h-full p-4 ${imageUrls.length > 1 ? 'grid grid-cols-1 md:grid-cols-2 gap-4 overflow-y-auto' : 'flex items-center justify-center'}`}>
                  {imageUrls.map((url, index) => (
                    <div key={index} className="relative group w-full h-full flex items-center justify-center bg-black/40 rounded-2xl overflow-hidden border border-blue-500/10 min-h-[300px]">
                      <img
                        src={url}
                        alt={`Generated result ${index + 1}`}
                        className={`object-contain ${imageUrls.length > 1 ? 'max-h-[400px]' : 'w-full h-full'}`}
                        referrerPolicy="no-referrer"
                      />
                      {/* Desktop Action Buttons (Hover Reveal) */}
                      <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-20 px-2 flex-wrap">
                        <button
                          onClick={() => handleUseAsInput(url)}
                          className="flex items-center gap-1.5 bg-emerald-600/80 hover:bg-emerald-500 backdrop-blur-xl border border-emerald-500/50 text-white px-3 py-2 rounded-full font-bold transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)] hover:scale-105 text-xs sm:text-sm"
                        >
                          <ImagePlus className="w-4 h-4" />
                          كمرجع
                        </button>
                        <button
                          onClick={() => handleUpscale(url)}
                          disabled={isUpscaling === url}
                          className="flex items-center gap-1.5 bg-purple-600/80 hover:bg-purple-500 backdrop-blur-xl border border-purple-500/50 text-white px-3 py-2 rounded-full font-bold transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)] hover:scale-105 text-xs sm:text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isUpscaling === url ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                          تكبير
                        </button>
                        <button
                          onClick={() => setSelectedImage(url)}
                          className="flex items-center gap-1.5 bg-black/60 hover:bg-blue-600/80 backdrop-blur-xl border border-blue-500/50 text-white px-3 py-2 rounded-full font-bold transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)] hover:scale-105 text-xs sm:text-sm"
                        >
                          <Maximize2 className="w-4 h-4" />
                          عرض
                        </button>
                        <button
                          onClick={() => openCropModal(url)}
                          className="flex items-center gap-1.5 bg-black/60 hover:bg-blue-600/80 backdrop-blur-xl border border-blue-500/50 text-white px-3 py-2 rounded-full font-bold transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)] hover:scale-105 text-xs sm:text-sm"
                        >
                          <Crop className="w-4 h-4" />
                          قص
                        </button>
                        <button
                          onClick={() => handleDownload(url)}
                          className="flex items-center gap-1.5 bg-blue-600/80 hover:bg-blue-500 backdrop-blur-xl border border-blue-400/50 text-white px-3 py-2 rounded-full font-bold transition-all shadow-[0_8px_30px_rgb(0,0,0,0.5)] hover:scale-105 text-xs sm:text-sm"
                        >
                          <Download className="w-4 h-4" />
                          تنزيل
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                /* Empty State */
                <div className="text-center text-blue-500/40 flex flex-col items-center gap-5 p-8">
                  <div className="w-24 h-24 rounded-full bg-blue-500/5 flex items-center justify-center mb-2 shadow-inner border border-blue-500/10">
                    <ImageIcon className="w-12 h-12 opacity-50" />
                  </div>
                  <p className="text-xl font-medium text-blue-200/70">مساحة العرض</p>
                  <p className="text-sm opacity-60 max-w-xs leading-relaxed text-blue-200/50">
                    أدخل وصفاً واضغط على زر التوليد لترى السحر يتحقق هنا
                  </p>
                </div>
              )}
            </div>
            
            {/* Mobile Action Buttons (Always visible on small screens when image exists) */}
            {imageUrls.length === 1 && (
              <div className="mt-6 lg:hidden grid grid-cols-2 sm:grid-cols-5 gap-3">
                <button
                  onClick={() => handleUseAsInput(imageUrls[0])}
                  className="w-full flex items-center justify-center gap-2 bg-emerald-600/40 hover:bg-emerald-600/60 backdrop-blur-md border border-emerald-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm"
                >
                  <ImagePlus className="w-4 h-4" />
                  كمرجع
                </button>
                <button
                  onClick={() => handleUpscale(imageUrls[0])}
                  disabled={isUpscaling === imageUrls[0]}
                  className="w-full flex items-center justify-center gap-2 bg-purple-600/40 hover:bg-purple-600/60 backdrop-blur-md border border-purple-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm disabled:opacity-50"
                >
                  {isUpscaling === imageUrls[0] ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                  تكبير
                </button>
                <button
                  onClick={() => setSelectedImage(imageUrls[0])}
                  className="w-full flex items-center justify-center gap-2 bg-black/40 hover:bg-black/60 backdrop-blur-md border border-blue-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm"
                >
                  <Maximize2 className="w-4 h-4" />
                  عرض
                </button>
                <button
                  onClick={() => openCropModal(imageUrls[0])}
                  className="w-full flex items-center justify-center gap-2 bg-black/40 hover:bg-black/60 backdrop-blur-md border border-blue-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm"
                >
                  <Crop className="w-4 h-4" />
                  قص
                </button>
                <button
                  onClick={() => handleDownload(imageUrls[0])}
                  className="w-full flex items-center justify-center gap-2 bg-blue-600/20 hover:bg-blue-600/40 backdrop-blur-md border border-blue-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm"
                >
                  <Download className="w-4 h-4" />
                  تنزيل
                </button>
                <button
                  onClick={() => openTemplateModal(imageUrls[0])}
                  className="w-full flex items-center justify-center gap-2 bg-indigo-600/20 hover:bg-indigo-600/40 backdrop-blur-md border border-indigo-500/30 text-white px-3 py-3 rounded-2xl font-bold transition-all active:scale-95 text-sm"
                >
                  <LayoutTemplate className="w-4 h-4" />
                  في قالب
                </button>
              </div>
            )}
          </div>

        </div>

        {/* History Section */}
        {history.length > 0 && (
          <div className="bg-black/40 backdrop-blur-xl border border-blue-500/20 p-6 md:p-10 rounded-[2.5rem] shadow-2xl mt-10">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Clock className="w-6 h-6 text-blue-400" />
                سجل الصور السابقة
              </h2>
              <button 
                onClick={() => setHistory([])} 
                className="text-red-400 hover:text-red-300 bg-red-500/10 hover:bg-red-500/20 px-4 py-2 rounded-xl flex items-center gap-2 text-sm transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                مسح السجل
              </button>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {history.map((item) => (
                <div key={item.id} className="relative group rounded-2xl overflow-hidden border border-blue-500/20 aspect-square bg-black/60 shadow-lg">
                  <img 
                    src={item.url} 
                    alt={item.prompt} 
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                    <p className="text-xs text-blue-100 line-clamp-3 mb-3 leading-relaxed" dir="rtl">{item.prompt}</p>
                    <div className="flex justify-center gap-2 flex-wrap">
                      <button 
                        onClick={() => handleUseAsInput(item.url)} 
                        className="bg-emerald-600/80 hover:bg-emerald-500 text-white p-2.5 rounded-full backdrop-blur-md transition-colors"
                        title="استخدام كمرجع"
                      >
                        <ImagePlus className="w-4 h-4"/>
                      </button>
                      <button 
                        onClick={() => handleUpscale(item.url)} 
                        disabled={isUpscaling === item.url}
                        className="bg-purple-600/80 hover:bg-purple-500 text-white p-2.5 rounded-full backdrop-blur-md transition-colors disabled:opacity-50"
                        title="تكبير الدقة"
                      >
                        {isUpscaling === item.url ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4"/>}
                      </button>
                      <button 
                        onClick={() => setSelectedImage(item.url)} 
                        className="bg-blue-600/80 hover:bg-blue-500 text-white p-2.5 rounded-full backdrop-blur-md transition-colors"
                        title="عرض"
                      >
                        <Maximize2 className="w-4 h-4"/>
                      </button>
                      <button 
                        onClick={() => openCropModal(item.url)} 
                        className="bg-gray-700/80 hover:bg-gray-600 text-white p-2.5 rounded-full backdrop-blur-md transition-colors"
                        title="قص"
                      >
                        <Crop className="w-4 h-4"/>
                      </button>
                      <button 
                        onClick={() => handleDownload(item.url)} 
                        className="bg-blue-600/80 hover:bg-blue-500 text-white p-2.5 rounded-full backdrop-blur-md transition-colors"
                        title="تنزيل"
                      >
                        <Download className="w-4 h-4"/>
                      </button>
                      <button 
                        onClick={() => openTemplateModal(item.url)} 
                        className="bg-indigo-600/80 hover:bg-indigo-500 text-white p-2.5 rounded-full backdrop-blur-md transition-colors"
                        title="تنزيل في قالب"
                      >
                        <LayoutTemplate className="w-4 h-4"/>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Prompt Bank Modal */}
      {isPromptBankOpen && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 sm:p-8" onClick={() => setIsPromptBankOpen(false)}>
          <div 
            className="bg-gray-900 border border-blue-500/30 rounded-3xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-800 bg-black/20">
              <div className="flex items-center gap-4">
                <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                  <Library className="w-6 h-6 text-blue-400" />
                  بنك الأوصاف (Prompt Bank)
                </h3>
                <button
                  onClick={() => setIsPromptBankEditMode(!isPromptBankEditMode)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    isPromptBankEditMode 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  <Edit2 className="w-4 h-4" />
                  {isPromptBankEditMode ? 'إنهاء التعديل' : 'تعديل'}
                </button>
              </div>
              <button 
                onClick={() => setIsPromptBankOpen(false)}
                className="text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 rounded-full p-2 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Content */}
            <div className="overflow-y-auto p-6 space-y-8 custom-scrollbar">
              {promptBank.map((category, idx) => (
                <div key={idx} className="space-y-4">
                  <div className="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h4 className="text-lg font-semibold text-blue-300 flex items-center gap-2">
                      {category.category}
                    </h4>
                    {isPromptBankEditMode && (
                      <button
                        onClick={() => handleDeleteCategory(idx)}
                        className="text-red-400 hover:text-red-300 p-1.5 rounded-lg hover:bg-red-500/10 transition-colors"
                        title="حذف التصنيف"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {category.prompts.map((p, pIdx) => (
                      <div key={pIdx} className="relative group flex flex-col">
                        <button
                          onClick={() => {
                            if (isPromptBankEditMode) return;
                            setPrompt(p.prompt);
                            setIsPromptBankOpen(false);
                          }}
                          className={`flex flex-col items-start text-right p-4 rounded-2xl bg-gray-800/50 border transition-all h-full ${
                            isPromptBankEditMode 
                              ? 'border-gray-700 cursor-default' 
                              : 'hover:bg-blue-900/30 border-gray-700 hover:border-blue-500/50 cursor-pointer'
                          }`}
                        >
                          <span className="font-bold text-gray-200 group-hover:text-blue-300 mb-2 pr-8">{p.title}</span>
                          <span className="text-xs text-gray-400 line-clamp-3 leading-relaxed" dir="ltr">{p.prompt}</span>
                        </button>
                        
                        {isPromptBankEditMode && (
                          <div className="absolute top-3 right-3 flex items-center gap-1">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingPrompt({ catIdx: idx, promptIdx: pIdx, title: p.title, prompt: p.prompt });
                              }}
                              className="p-1.5 text-gray-400 hover:text-blue-400 bg-gray-900 rounded-lg border border-gray-700 hover:border-blue-500/50 transition-colors"
                            >
                              <Edit2 className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeletePrompt(idx, pIdx);
                              }}
                              className="p-1.5 text-gray-400 hover:text-red-400 bg-gray-900 rounded-lg border border-gray-700 hover:border-red-500/50 transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {/* Add Prompt Button */}
                    {isPromptBankEditMode && (
                      <button
                        onClick={() => setEditingPrompt({ catIdx: idx, promptIdx: -1, title: "", prompt: "" })}
                        className="flex flex-col items-center justify-center text-center p-4 rounded-2xl bg-gray-800/20 hover:bg-gray-800/50 border border-dashed border-gray-700 hover:border-blue-500/50 transition-all text-gray-400 hover:text-blue-400 min-h-[120px]"
                      >
                        <Plus className="w-6 h-6 mb-2" />
                        <span className="text-sm font-medium">إضافة وصف جديد</span>
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {/* Add Category Section */}
              {isPromptBankEditMode && (
                <div className="pt-6 border-t border-gray-800">
                  {isAddingCategory ? (
                    <div className="flex items-center gap-3 bg-gray-800/50 p-4 rounded-2xl border border-gray-700">
                      <input
                        type="text"
                        value={newCategoryName}
                        onChange={(e) => setNewCategoryName(e.target.value)}
                        placeholder="اسم التصنيف الجديد..."
                        className="flex-1 bg-black/50 border border-gray-600 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                        autoFocus
                      />
                      <button
                        onClick={handleAddCategory}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-xl font-medium transition-colors"
                      >
                        إضافة
                      </button>
                      <button
                        onClick={() => {
                          setIsAddingCategory(false);
                          setNewCategoryName("");
                        }}
                        className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-xl font-medium transition-colors"
                      >
                        إلغاء
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setIsAddingCategory(true)}
                      className="flex items-center justify-center gap-2 w-full p-4 rounded-2xl bg-gray-800/20 hover:bg-gray-800/50 border border-dashed border-gray-700 hover:border-blue-500/50 transition-all text-gray-400 hover:text-blue-400"
                    >
                      <Plus className="w-5 h-5" />
                      <span className="font-medium">إضافة تصنيف جديد</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Edit/Add Prompt Modal */}
      {editingPrompt && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={() => setEditingPrompt(null)}>
          <div 
            className="bg-gray-900 border border-blue-500/30 rounded-3xl shadow-2xl w-full max-w-lg p-6 space-y-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-xl font-bold text-white flex items-center gap-2 border-b border-gray-800 pb-4">
              <Edit2 className="w-5 h-5 text-blue-400" />
              {editingPrompt.promptIdx === -1 ? 'إضافة وصف جديد' : 'تعديل الوصف'}
            </h3>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">عنوان الوصف</label>
                <input
                  type="text"
                  value={editingPrompt.title}
                  onChange={(e) => setEditingPrompt({...editingPrompt, title: e.target.value})}
                  placeholder="مثال: مدينة مستقبلية"
                  className="w-full bg-black/50 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">النص (Prompt)</label>
                <textarea
                  value={editingPrompt.prompt}
                  onChange={(e) => setEditingPrompt({...editingPrompt, prompt: e.target.value})}
                  placeholder="أدخل الوصف باللغة الإنجليزية..."
                  rows={5}
                  className="w-full bg-black/50 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500 resize-none"
                  dir="ltr"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-800">
              <button
                onClick={() => setEditingPrompt(null)}
                className="px-5 py-2.5 rounded-xl text-gray-300 hover:text-white hover:bg-gray-800 font-medium transition-colors"
              >
                إلغاء
              </button>
              <button
                onClick={handleSavePrompt}
                disabled={!editingPrompt.title.trim() || !editingPrompt.prompt.trim()}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-6 py-2.5 rounded-xl font-bold transition-colors"
              >
                <Save className="w-4 h-4" />
                حفظ
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Template Modal */}
      {isTemplateModalOpen && imageToTemplate && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4" onClick={() => setIsTemplateModalOpen(false)}>
          <div 
            className="bg-gray-900 border border-indigo-500/30 rounded-3xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-800 bg-black/20">
              <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                <LayoutTemplate className="w-6 h-6 text-indigo-400" />
                تنزيل بداخل القالب
              </h3>
              <button 
                onClick={() => setIsTemplateModalOpen(false)}
                className="text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 rounded-full p-2 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Content */}
            <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
              {/* Left/Top: Preview */}
              <div className="flex-1 p-6 bg-black/50 overflow-auto flex items-center justify-center relative">
                {!templateImage ? (
                  <div className="text-center space-y-4">
                    <div className="w-24 h-24 bg-gray-800 rounded-full flex items-center justify-center mx-auto border border-dashed border-gray-600">
                      <Upload className="w-10 h-10 text-gray-400" />
                    </div>
                    <p className="text-gray-400">الرجاء رفع صورة القالب أولاً</p>
                    <input 
                      type="file" 
                      accept="image/*" 
                      onChange={handleTemplateUpload} 
                      className="hidden" 
                      id="template-upload" 
                    />
                    <label 
                      htmlFor="template-upload" 
                      className="inline-block bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2.5 rounded-xl font-medium cursor-pointer transition-colors"
                    >
                      اختيار صورة القالب
                    </label>
                  </div>
                ) : (
                  <div className="relative shadow-2xl" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                    <canvas ref={templateCanvasRef} className="max-w-full max-h-[60vh] md:max-h-[70vh] object-contain" />
                  </div>
                )}
              </div>

              {/* Right/Bottom: Controls */}
              {templateImage && (
                <div className="w-full md:w-80 p-6 border-t md:border-t-0 md:border-r border-gray-800 bg-gray-900/50 overflow-y-auto space-y-6">
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Settings2 className="w-5 h-5 text-indigo-400" />
                      إعدادات الموضع
                    </h4>
                    
                    <div className="space-y-2">
                      <label className="flex justify-between text-sm text-gray-300">
                        <span>الحجم (Scale)</span>
                        <span>{templateSettings.scale}%</span>
                      </label>
                      <input 
                        type="range" 
                        min="10" max="200" 
                        value={templateSettings.scale} 
                        onChange={(e) => setTemplateSettings({...templateSettings, scale: parseInt(e.target.value)})}
                        className="w-full accent-indigo-500"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="flex justify-between text-sm text-gray-300">
                        <span>الموضع الأفقي (X)</span>
                        <span>{templateSettings.offsetX}%</span>
                      </label>
                      <input 
                        type="range" 
                        min="0" max="100" 
                        value={templateSettings.offsetX} 
                        onChange={(e) => setTemplateSettings({...templateSettings, offsetX: parseInt(e.target.value)})}
                        className="w-full accent-indigo-500"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="flex justify-between text-sm text-gray-300">
                        <span>الموضع الرأسي (Y)</span>
                        <span>{templateSettings.offsetY}%</span>
                      </label>
                      <input 
                        type="range" 
                        min="0" max="100" 
                        value={templateSettings.offsetY} 
                        onChange={(e) => setTemplateSettings({...templateSettings, offsetY: parseInt(e.target.value)})}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                  </div>

                  <div className="pt-4 border-t border-gray-800 space-y-3">
                    <button 
                      onClick={handleDownloadTemplate}
                      className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-3 rounded-xl font-bold transition-colors"
                    >
                      <Download className="w-5 h-5" />
                      تنزيل الصورة النهائية
                    </button>
                    
                    <div className="text-center">
                      <input 
                        type="file" 
                        accept="image/*" 
                        onChange={handleTemplateUpload} 
                        className="hidden" 
                        id="template-change" 
                      />
                      <label 
                        htmlFor="template-change" 
                        className="text-sm text-gray-400 hover:text-white cursor-pointer transition-colors"
                      >
                        تغيير القالب
                      </label>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
