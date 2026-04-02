import { createBrowserRouter } from "react-router";
import { LandingPage } from "@/app/components/LandingPage";
import { UploadPage } from "@/app/components/UploadPage";
import { ResultsPage } from "@/app/components/ResultsPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LandingPage,
  },
  {
    path: "/upload",
    Component: UploadPage,
  },
  {
    path: "/results",
    Component: ResultsPage,
  },
]);
