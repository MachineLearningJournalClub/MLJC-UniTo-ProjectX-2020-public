//___  ____       ___ _____   _   _       _ _
//|  \/  | |     |_  /  __ \ | | | |     (_| |
//| .  . | |       | | /  \/ | | | |_ __  _| |_ ___
//| |\/| | |       | | |     | | | | '_ \| | __/ _ \
//| |  | | |___/\__/ | \__/\ | |_| | | | | | || (_) |
//_______\_____\____/ \____/  _____|_|___|_____\_____ _____ _____
//| ___ \        (_)         | | \ \ / / / __  |  _  / __  |  _  |
//| |_/ _ __ ___  _  ___  ___| |_ \ V /  `' / /| |/' `' / /| |/' |
//|  __| '__/ _ \| |/ _ \/ __| __|/   \    / / |  /| | / / |  /| |
//| |  | | | (_) | |  __| (__| |_/ /^\ \ ./ /__\ |_/ ./ /__\ |_/ /
//\_|  |_|  \___/| |\___|\___|\__\/   \/ \_____/\___/\_____/\___/
//              _/ |
//             |__/
//
// This code is part of the proposal of the team "MLJC UniTo" - University of Turin
// for "ProjectX 2020" Climate Change for AI.
// The code is licensed under MIT 3.0
// Please read readme or comments for credits and further information.

// Compiler: CERN ROOT 6

// Short description of this file: CERN Root for surface fit.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

double fexample(double x, double y) {
        return x*x + y*y +x;
    }

void Surface_fit()
{
    std::vector<double> xaxis;
    std::vector<double> yaxis;

    int xresolution = 20;
    int yresolution = 20;
    double xlo = -3.;
    double xhi = 3.0;
    double ylo = -3.;
    double yhi = 3.0;

    //Creating linspaces
    for (double l = xlo; l < xhi; l = l + ((xhi-xlo)/xresolution)) {
        xaxis.push_back(l);
    }

    for (double l = ylo; l < yhi; l = l + ((yhi-ylo)/yresolution)) {
        yaxis.push_back(l);
    }

    //Create the TH2D object:
    TH2D * h2 = new TH2D("h2", "Interpolation", xresolution, xlo, xhi, yresolution, ylo, yhi);

    //Generate example data, replace the following lines with the import commands
    for (int i = 0; i < xaxis.size(); ++i)
    {
        for (int j = 0; j < yaxis.size(); ++j)
        {
            Int_t binx = h2->GetXaxis()->FindBin(xaxis[i]);
            Int_t biny = h2->GetXaxis()->FindBin(yaxis[j]);
            h2->SetBinContent(binx, biny, fexample(xaxis[i], yaxis[j]));
            std::cout << "\nFilling in \t" << xaxis[i] << " \t" << xaxis[j] << "\t value \t" << fexample(xaxis[i], yaxis[j]);
        }
    }

    //Try to fit with 2 deg poly
    TF2 *f2 = new TF2("f2","[0] + [1]*x + [2]*y + [3]*x*x + [4]*y*y",0,5,0,5);
    f2->SetParameters(0., 2, 2, 2, 2);
    h2->Fit(f2);

    cout << "\nThe symbolic intrpolation is \n" << "z=" <<
         f2->GetFormula()->GetExpFormula() << "\nWith parameters listed above\n\n";

    h2->Draw("lego2");

}
