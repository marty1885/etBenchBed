// perf_surface.C - Plot GFLOPS as a 2D surface with one dimension fixed
// Usage: root -l 'analysis/perf_surface.C("N", 16, "mm_sweep.csv")'     -- fix N=16, plot GFLOPS(M, K)
//        root -l 'analysis/perf_surface.C("M", 4096, "mm_sweep.csv")'   -- fix M=4096, plot GFLOPS(K, N)
//        root -l 'analysis/perf_surface.C("K", 4096, "mm_sweep.csv")'   -- fix K=4096, plot GFLOPS(M, N)

void perf_surface(const char *fixDim = "N", int fixVal = 16,
                  const char *csvFile = "mul_mat_f32_perf.csv") {

    TString fix(fixDim);
    fix.ToUpper();

    auto df = ROOT::RDF::FromCSV(csvFile);

    auto filtered = df.Filter([&](Long64_t M, Long64_t K, Long64_t N) {
        if (fix == "M") return M == (Long64_t)fixVal;
        if (fix == "K") return K == (Long64_t)fixVal;
        return N == (Long64_t)fixVal;
    }, {"M", "K", "N"});

    std::string xCol, yCol;
    if      (fix == "M") { xCol = "K"; yCol = "N"; }
    else if (fix == "K") { xCol = "M"; yCol = "N"; }
    else                 { xCol = "M"; yCol = "K"; }

    auto xVec = filtered.Take<Long64_t>(xCol);
    auto yVec = filtered.Take<Long64_t>(yCol);
    auto zVec = filtered.Take<double>("GFLOPS");

    auto &xv = xVec.GetValue();
    auto &yv = yVec.GetValue();
    auto &zv = zVec.GetValue();

    int npts = xv.size();
    if (npts == 0) {
        printf("No data found for %s = %d\n", fixDim, fixVal);
        return;
    }
    printf("Found %d points with %s = %d\n", npts, fixDim, fixVal);

    auto *gr = new TGraph2D(npts);
    gr->SetName("perfSurface");
    gr->SetTitle(Form("f32 mul_mat GFLOPS  (%s = %d);%s;%s;GFLOPS",
                       fixDim, fixVal, xCol.c_str(), yCol.c_str()));

    for (int i = 0; i < npts; i++) {
        gr->SetPoint(i, xv[i], yv[i], zv[i]);
    }

    const int scale = 3;
    const int baseW = 900, baseH = 700;

    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBird);
    gStyle->SetNumberContours(40);

    auto *c = new TCanvas("c1",
        Form("Performance Surface (%s=%d)", fixDim, fixVal),
        baseW * scale, baseH * scale);
    c->SetTheta(30);
    c->SetPhi(45);
    c->SetLeftMargin(0.14);
    c->SetRightMargin(0.16);

    gr->SetNpx(200);
    gr->SetNpy(200);
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(0.6 * scale);
    gr->Draw("surf1z");

    c->Update();

    TString hiresName = Form("/tmp/perf_surface_%s%d_hires.png", fixDim, fixVal);
    c->SaveAs(hiresName);

    TString outName = Form("perf_surface_%s%d.png", fixDim, fixVal);
    auto *img = TImage::Open(hiresName);
    img->Scale(baseW, baseH);
    img->WriteImage(outName);
    delete img;
    printf("Saved %s (%dx SSAA)\n", outName.Data(), scale);
}
