import { ArrowLeft, Activity } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { useNavigate, useLocation } from "react-router";

export function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state || {};
  
  const resultImage = state.resultImage;
  const beforeImage = state.beforeImage;
  const afterImage = state.afterImage;
  const actualImage = state.actualImage;
  const stats = state.stats;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-6 flex flex-col items-center">
      <div className="w-full max-w-6xl">
        <div className="flex items-center justify-between mb-8">
          <Button variant="ghost" onClick={() => navigate("/upload")} className="text-zinc-400 hover:text-zinc-100">
            <ArrowLeft className="size-4 mr-2" />
            Back to Upload
          </Button>
          <h1 className="text-2xl font-semibold tracking-tight">Change Detection Analysis</h1>
          <div className="w-[120px]"></div> {/* spacer to center title */}
        </div>

        {stats && (
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6 mb-8 flex items-center gap-6">
            <div className="bg-cyan-500/10 p-4 rounded-full">
              <Activity className="size-8 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg text-zinc-400 mb-1">Total Detected Change</h2>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-bold tracking-tight text-white">
                  {stats.percentage.toFixed(3)}%
                </span>
                <span className="text-zinc-500">({stats.pixels} pixels changed)</span>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {beforeImage && (
            <div className="space-y-3">
              <h3 className="text-lg text-zinc-300 font-medium">Before (Time T1)</h3>
              <img src={beforeImage} className="w-full rounded-lg border border-zinc-700 bg-black aspect-square object-cover" />
            </div>
          )}
          {afterImage && (
            <div className="space-y-3">
              <h3 className="text-lg text-zinc-300 font-medium">After (Time T2)</h3>
              <img src={afterImage} className="w-full rounded-lg border border-zinc-700 bg-black aspect-square object-cover" />
            </div>
          )}
          {actualImage && (
            <div className="space-y-3">
              <h3 className="text-lg text-zinc-300 font-medium">Ground Truth (Actual Outcome)</h3>
              <img src={actualImage} className="w-full rounded-lg border border-zinc-700 bg-black aspect-square object-cover" />
            </div>
          )}
          {resultImage && (
            <div className="space-y-3">
              <h3 className="text-lg text-cyan-400 font-medium">Predicted Change Mask</h3>
              <img src={resultImage} className="w-full rounded-lg border-2 border-cyan-800/50 bg-black aspect-square object-cover" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
