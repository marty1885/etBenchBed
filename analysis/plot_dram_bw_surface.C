#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TAxis.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"
#include "TROOT.h"
#include "TStyle.h"

namespace {

struct Row {
  int m = 0;
  int k = 0;
  int n = 0;
  double bw_gb_s = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

std::vector<double> build_edges(const std::vector<int>& values) {
  if (values.empty()) {
    throw std::runtime_error("cannot build histogram edges from an empty axis");
  }

  if (values.size() == 1) {
    const double center = values.front();
    return {center - 0.5, center + 0.5};
  }

  std::vector<double> edges(values.size() + 1, 0.0);
  edges.front() = values[0] - 0.5 * (values[1] - values[0]);
  for (std::size_t i = 1; i < values.size(); ++i) {
    edges[i] = 0.5 * (values[i - 1] + values[i]);
  }
  edges.back() = values.back() + 0.5 * (values.back() - values[values.size() - 2]);
  return edges;
}

}  // namespace

void plot_dram_bw_surface(const char* csv_path = "dram_bw_sweep_pure_copy.csv",
                          const char* output_path = "dram_bw_surface.png",
                          bool save_output = false,
                          int fit_n_min = 128,
                          int fit_n_max = 8192) {
  std::ifstream input(csv_path);
  if (!input.is_open()) {
    std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
    return;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line[0] != '#') break;
  }
  if (input.fail() && !input.eof()) {
    std::cerr << "CSV file is empty: " << csv_path << std::endl;
    return;
  }

  const auto header = split_csv_line(line);
  if (header.size() < 6 || header[0] != "M" || header[1] != "K" || header[2] != "N" ||
      header[5] != "BW_GB_S") {
    std::cerr << "Unexpected CSV header in " << csv_path << std::endl;
    return;
  }

  std::vector<Row> rows;
  std::set<int> m_values;
  std::set<int> n_values;
  std::map<int, std::vector<Row>> rows_by_n;
  int skipped_non_square = 0;

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    const auto fields = split_csv_line(line);
    if (fields.size() < 6) {
      continue;
    }

    Row row;
    row.m = std::stoi(fields[0]);
    row.k = std::stoi(fields[1]);
    row.n = std::stoi(fields[2]);
    row.bw_gb_s = std::stod(fields[5]);

    if (row.m != row.k) {
      ++skipped_non_square;
      continue;
    }

    rows.push_back(row);
    m_values.insert(row.m);
    n_values.insert(row.n);
    rows_by_n[row.n].push_back(row);
  }

  if (rows.empty()) {
    std::cerr << "No rows with M == K were found in " << csv_path << std::endl;
    return;
  }

  std::vector<int> m_axis(m_values.begin(), m_values.end());
  std::vector<int> n_axis(n_values.begin(), n_values.end());
  const std::vector<double> m_edges = build_edges(m_axis);
  const std::vector<double> n_edges = build_edges(n_axis);

  TH2D* surface = new TH2D("dram_bw_surface",
                           "DRAM pure-copy bandwidth surface;N;M (= K);Bandwidth (GB/s)",
                           static_cast<int>(n_axis.size()), n_edges.data(),
                           static_cast<int>(m_axis.size()), m_edges.data());
  surface->SetStats(false);
  surface->SetContour(99);

  std::map<int, int> m_bin;
  std::map<int, int> n_bin;
  for (std::size_t i = 0; i < m_axis.size(); ++i) {
    m_bin[m_axis[i]] = static_cast<int>(i) + 1;
  }
  for (std::size_t i = 0; i < n_axis.size(); ++i) {
    n_bin[n_axis[i]] = static_cast<int>(i) + 1;
  }

  for (const auto& row : rows) {
    surface->SetBinContent(n_bin[row.n], m_bin[row.m], row.bw_gb_s);
  }

  std::vector<double> peak_n;
  std::vector<double> peak_m;
  std::vector<double> peak_bw;
  peak_n.reserve(rows_by_n.size());
  peak_m.reserve(rows_by_n.size());
  peak_bw.reserve(rows_by_n.size());

  for (const auto& [n, n_rows] : rows_by_n) {
    const Row* best = nullptr;
    for (const auto& row : n_rows) {
      if (best == nullptr || row.bw_gb_s > best->bw_gb_s) {
        best = &row;
      }
    }
    if (best != nullptr) {
      peak_n.push_back(static_cast<double>(best->n));
      peak_m.push_back(static_cast<double>(best->m));
      peak_bw.push_back(best->bw_gb_s);
    }
  }

  const double ridge_z = *std::max_element(peak_bw.begin(), peak_bw.end());
  std::vector<double> fit_peak_n;
  std::vector<double> fit_peak_m;
  for (std::size_t i = 0; i < peak_n.size(); ++i) {
    if (peak_n[i] >= fit_n_min && peak_n[i] <= fit_n_max) {
      fit_peak_n.push_back(peak_n[i]);
      fit_peak_m.push_back(peak_m[i]);
    }
  }

  if (fit_peak_n.size() < 2) {
    std::cerr << "Not enough peak points in requested fit range [" << fit_n_min << ", " << fit_n_max
              << "]" << std::endl;
    return;
  }

  gROOT->SetBatch(kFALSE);
  gStyle->SetPalette(kViridis);
  gStyle->SetNumberContours(99);
  gStyle->SetOptStat(0);
  gStyle->SetPadRightMargin(0.16);
  gStyle->SetPadLeftMargin(0.10);
  gStyle->SetPadBottomMargin(0.12);

  TCanvas* canvas = new TCanvas("c_dram_bw_surface", "DRAM BW Surface", 1400, 900);
  canvas->SetTheta(26.0);
  canvas->SetPhi(242.0);
  canvas->SetLeftMargin(0.10);
  canvas->SetRightMargin(0.18);
  canvas->SetBottomMargin(0.12);

  surface->GetXaxis()->CenterTitle();
  surface->GetYaxis()->CenterTitle();
  surface->GetZaxis()->CenterTitle();
  surface->GetXaxis()->SetTitleOffset(1.25);
  surface->GetYaxis()->SetTitleOffset(1.45);
  surface->GetZaxis()->SetTitleOffset(1.35);

  surface->Draw("SURF1Z");

  TPolyMarker3D* ridge_points = new TPolyMarker3D(static_cast<int>(peak_n.size()));
  ridge_points->SetMarkerStyle(29);
  ridge_points->SetMarkerSize(1.4);
  ridge_points->SetMarkerColor(kMagenta + 2);
  for (std::size_t i = 0; i < peak_n.size(); ++i) {
    ridge_points->SetPoint(static_cast<int>(i), peak_n[i], peak_m[i], peak_bw[i]);
  }
  ridge_points->Draw();

  TGraph* ridge_graph = new TGraph(static_cast<int>(fit_peak_n.size()), fit_peak_n.data(), fit_peak_m.data());

  struct FitSpec {
    const char* name;
    const char* formula;
    const char* legend_label;
    std::vector<double> init;
    Color_t color;
    int width;
  };

  std::vector<FitSpec> top_fits = {
      {"rational", "([0] + [1]*x)/(1 + [2]*x)", "([0] + [1]*N) / (1 + [2]*N)",
       {2400.0, -0.2, 1.0e-4}, kOrange + 7, 5},
      {"quadratic", "[0] + [1]*x + [2]*x*x", "[0] + [1]*N + [2]*N^{2}",
       {2500.0, -0.25, -1.0e-5}, kRed + 1, 4},
      {"log2_quad", "[0] + [1]*(log(x)/log(2.0)) + [2]*pow(log(x)/log(2.0),2)",
       "[0] + [1]*log_{2}(N) + [2]*log_{2}(N)^{2}",
       {-1000.0, 900.0, -60.0}, kCyan + 2, 4},
  };

  std::vector<TF1*> fitted_functions;
  std::vector<TPolyLine3D*> fit_lines;
  fitted_functions.reserve(top_fits.size());
  fit_lines.reserve(top_fits.size());

  TLegend* legend = new TLegend(0.64, 0.70, 0.92, 0.90);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  TLegendEntry* peak_entry = legend->AddEntry(ridge_points, "Peak points", "p");
  peak_entry->SetMarkerColor(kMagenta + 2);
  peak_entry->SetMarkerStyle(29);
  peak_entry->SetMarkerSize(1.4);

  for (std::size_t ifit = 0; ifit < top_fits.size(); ++ifit) {
    const auto& spec = top_fits[ifit];
    TF1* fit = new TF1(Form("peak_fit_surface_%zu", ifit), spec.formula, fit_peak_n.front(), fit_peak_n.back());
    for (std::size_t ip = 0; ip < spec.init.size(); ++ip) {
      fit->SetParameter(static_cast<int>(ip), spec.init[ip]);
    }
    ridge_graph->Fit(fit, "RQ");
    fitted_functions.push_back(fit);

    TPolyLine3D* fit_line = new TPolyLine3D(static_cast<int>(peak_n.size()));
    fit_line->SetLineColor(spec.color);
    fit_line->SetLineWidth(spec.width);
    for (std::size_t i = 0; i < peak_n.size(); ++i) {
      const double n = peak_n[i];
      const double fitted_m = fit->Eval(n);
      fit_line->SetPoint(static_cast<int>(i), n, fitted_m, ridge_z - 0.15 * static_cast<double>(ifit));
    }
    fit_line->Draw();
    fit_lines.push_back(fit_line);
    TLegendEntry* line_entry = legend->AddEntry(fit_line, spec.legend_label, "l");
    line_entry->SetLineColor(spec.color);
    line_entry->SetLineWidth(spec.width);
  }

  legend->Draw();

  canvas->Modified();
  canvas->Update();

  if (save_output) {
    canvas->SaveAs(output_path);
  }

  std::cout << "Loaded " << rows.size() << " rows with M == K";
  if (skipped_non_square > 0) {
    std::cout << " (" << skipped_non_square << " non-square rows skipped)";
  }
  std::cout << std::endl;
  if (save_output) {
    std::cout << "Wrote " << output_path << std::endl;
  } else {
    std::cout << "Canvas remains open in the ROOT session" << std::endl;
  }
  std::cout << "Top fits on N in [" << fit_n_min << ", " << fit_n_max << "]:" << std::endl;
  for (std::size_t ifit = 0; ifit < top_fits.size(); ++ifit) {
    std::cout << "  " << top_fits[ifit].name << ":";
    for (int ip = 0; ip < fitted_functions[ifit]->GetNpar(); ++ip) {
      std::cout << " p" << ip << "=" << fitted_functions[ifit]->GetParameter(ip);
    }
    std::cout << std::endl;
  }
}
