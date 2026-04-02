import { useNavigate } from "react-router";
import { Satellite, Shield, AlertTriangle } from "lucide-react";
import { Button } from "@/app/components/ui/button";

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-950/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <Satellite className="size-6 text-cyan-500" />
            <h1 className="text-xl tracking-tight">SATELLITE CHANGE DETECTION SYSTEM</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="max-w-4xl w-full">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center size-20 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
              <Shield className="size-10 text-cyan-500" />
            </div>
            <h2 className="text-4xl mb-4 tracking-tight">
              AI-Powered Satellite Image Analysis
            </h2>
            <p className="text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed">
              Advanced change detection system for security monitoring and intelligence analysis
            </p>
          </div>

          {/* Feature Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <div className="size-12 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4">
                <Satellite className="size-6 text-cyan-500" />
              </div>
              <h3 className="mb-2">Temporal Analysis</h3>
              <p className="text-sm text-zinc-400">
                Compare satellite imagery across different time periods to identify changes in terrain, infrastructure, and activity.
              </p>
            </div>

            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <div className="size-12 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4">
                <Shield className="size-6 text-cyan-500" />
              </div>
              <h3 className="mb-2">Security Monitoring</h3>
              <p className="text-sm text-zinc-400">
                Real-time detection of structural changes, movement patterns, and anomalies for enhanced situational awareness.
              </p>
            </div>

            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <div className="size-12 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4">
                <AlertTriangle className="size-6 text-cyan-500" />
              </div>
              <h3 className="mb-2">Precision Detection</h3>
              <p className="text-sm text-zinc-400">
                High-accuracy AI algorithms identify changes with detailed area calculations and visualization overlays.
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="text-center">
            <Button
              onClick={() => navigate("/upload")}
              className="bg-cyan-600 hover:bg-cyan-700 text-white px-8 py-6 text-lg h-auto"
            >
              Begin Analysis
            </Button>
          </div>

          {/* Ethics Disclaimer */}
          <div className="mt-16 bg-amber-500/5 border border-amber-500/20 rounded-lg p-6">
            <div className="flex gap-4">
              <AlertTriangle className="size-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-amber-500 mb-2">Ethics & Human Oversight</h4>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  This system is designed to assist human analysts in their decision-making process and does not make autonomous decisions. 
                  All detections require human verification and contextual analysis. The system serves as a support tool to enhance, 
                  not replace, expert human judgment in security and intelligence operations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 py-6">
        <div className="container mx-auto px-6 text-center text-sm text-zinc-500">
          CLASSIFIED SYSTEM • AUTHORIZED PERSONNEL ONLY
        </div>
      </footer>
    </div>
  );
}
