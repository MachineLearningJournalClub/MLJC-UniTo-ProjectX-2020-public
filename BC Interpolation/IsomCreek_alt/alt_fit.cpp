//Please run using CERN ROOT 6

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>


void alt_fit()
{
    int xresolution = 91;
    int yresolution = 91;
    double xlo = 0.;
    double xhi = 10.;
    double ylo = 0.;
    double yhi = 10.;

    //Create the TH2D object:
    TH2D * h2 = new TH2D("h2", "Interpolation", xresolution, xlo, xhi, yresolution, ylo, yhi);

    std::ifstream fin("./altMatrixLinearized.txt");
    
    for (int i = 0; i < xresolution; ++i)
    {
    	for (int j = 0; j < yresolution; ++j)
    	{
    		double d;
    		fin >> d;
    		h2->SetBinContent(i, j, d);
    	}
    }



   TF2 *f2 = new TF2("f2", "[p0]+[p1]*x+[p2]*x*x+[p3]*x*x*x+[p4]*x*x*x*x+[p5]*x*x*x*x*x + [p6]*y + [p7]*y*y + [p8]*y*y*y+[p9]*y*y*y*y+[p10]*y*y*y*y*y+[p11]*x*y+[p12]*x*x*y+[p13]*x*y*y + [p14]*x*x*y*y" ,0.,10.,0.,10.);

    //FUMILI Package (good for high number of parameters) https://www.sciencedirect.com/science/article/pii/S0010465513002622
    h2->Fit(f2);

    cout << "\nThe symbolic intrpolation is \n" << "z=" <<
    f2->GetFormula()->GetExpFormula() << "\nWith parameters listed above\n\n";
    
    h2->Draw("lego2");

}
