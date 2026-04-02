import { useState, useRef } from "react";
import { useNavigate } from "react-router";
import { Satellite, Upload, ArrowLeft, AlertCircle } from "lucide-react";
import { Button } from "@/app/components/ui/button";

export function UploadPage() {
  const navigate = useNavigate();

  // Preview images (Base64)
  const [beforeImage, setBeforeImage] = useState<string | null>(null);
  const [afterImage, setAfterImage] = useState<string | null>(null);

  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [actualFile, setActualFile] = useState<File | null>(null);
  const [actualImage, setActualImage] = useState<string | null>(null);

  const [status, setStatus] = useState<string>("");

  const beforeInputRef = useRef<HTMLInputElement>(null);
  const afterInputRef = useRef<HTMLInputElement>(null);
  const actualInputRef = useRef<HTMLInputElement>(null);

  const compressImage = (file: File, callback: (base64: string) => void) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target?.result as string;
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const MAX_WIDTH = 800;
        const MAX_HEIGHT = 800;
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
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(img, 0, 0, width, height);
        // Compress to 80% quality JPEG to save massive amounts of space
        const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
        callback(dataUrl);
      };
    };
  };

  const handleBeforeUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setBeforeFile(file);
      compressImage(file, setBeforeImage);
    }
  };

  const handleAfterUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAfterFile(file);
      compressImage(file, setAfterImage);
    }
  };

  const handleActualUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setActualFile(file);
      compressImage(file, setActualImage);
    }
  };

const handleAnalyze = async () => {
  if (!beforeInputRef.current?.files || !afterInputRef.current?.files) return;

  const formData = new FormData();
  formData.append("before_image", beforeInputRef.current.files[0]);
  formData.append("after_image", afterInputRef.current.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:8000/detect-change", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      alert("Backend error");
      return;
    }

    const data = await response.json();
    const imageUrl = `data:image/png;base64,${data.mask_base64}`;

    const statePayload = {
      resultImage: imageUrl,
      beforeImage: beforeImage,
      afterImage: afterImage,
      actualImage: actualImage,
      stats: {
        percentage: data.change_percentage,
        pixels: data.change_pixels
      }
    };

    navigate("/results", { state: statePayload });
  } catch (err: any) {
    console.error(err);
    alert(`Error running analysis: ${err.message || "Failed to connect to backend"}`);
  }
};



  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-950/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Satellite className="size-6 text-cyan-500" />
              <h1 className="text-xl tracking-tight">
                SATELLITE CHANGE DETECTION SYSTEM
              </h1>
            </div>
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="text-zinc-400 hover:text-zinc-100"
            >
              <ArrowLeft className="size-4 mr-2" />
              Back
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-6 py-12">
        <div className="max-w-5xl mx-auto">
          <div className="mb-8">
            <h2 className="text-3xl mb-2 tracking-tight">
              Upload Satellite Images
            </h2>
            <p className="text-zinc-400">
              Upload two satellite images from different time periods for change detection analysis
            </p>
          </div>

          {/* Upload Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {/* Before Image */}
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <h3 className="mb-2">Before Image (Time T1)</h3>

              <input
                ref={beforeInputRef}
                type="file"
                accept="image/*"
                onChange={handleBeforeUpload}
                className="hidden"
              />

              {beforeImage ? (
                <img
                  src={beforeImage}
                  className="rounded-lg border border-zinc-700"
                />
              ) : (
                <button
                  onClick={() => beforeInputRef.current?.click()}
                  className="w-full aspect-video border-2 border-dashed border-zinc-700 flex items-center justify-center hover:border-cyan-500 hover:text-cyan-500 transition-colors"
                >
                  <Upload />
                </button>
              )}
            </div>

            {/* After Image */}
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <h3 className="mb-2">After Image (Time T2)</h3>

              <input
                ref={afterInputRef}
                type="file"
                accept="image/*"
                onChange={handleAfterUpload}
                className="hidden"
              />

              {afterImage ? (
                <img
                  src={afterImage}
                  className="rounded-lg border border-zinc-700"
                />
              ) : (
                <button
                  onClick={() => afterInputRef.current?.click()}
                  className="w-full aspect-video border-2 border-dashed border-zinc-700 flex items-center justify-center hover:border-cyan-500 hover:text-cyan-500 transition-colors"
                >
                  <Upload />
                </button>
              )}
            </div>

            {/* Actual Output (Ground Truth) */}
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <h3 className="mb-2">Ground Truth (Optional)</h3>

              <input
                ref={actualInputRef}
                type="file"
                accept="image/*"
                onChange={handleActualUpload}
                className="hidden"
              />

              {actualImage ? (
                <img
                  src={actualImage}
                  className="rounded-lg border border-zinc-700"
                />
              ) : (
                <button
                  onClick={() => actualInputRef.current?.click()}
                  className="w-full aspect-video border-2 border-dashed border-zinc-700 flex items-center justify-center hover:border-cyan-500 hover:text-cyan-500 transition-colors text-zinc-500 flex-col gap-2"
                >
                  <Upload />
                  <span className="text-sm">For comparison</span>
                </button>
              )}
            </div>
          </div>

          {/* Status */}
          {status && (
            <p className="text-center text-sm text-cyan-400 mb-6">
              {status}
            </p>
          )}

          {/* Action Button */}
          <div className="flex justify-center">
            <Button
              onClick={handleAnalyze}
              disabled={!beforeFile || !afterFile}
              className="bg-cyan-600 hover:bg-cyan-700 px-12 py-6 text-lg"
            >
              Run Change Detection Analysis
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
