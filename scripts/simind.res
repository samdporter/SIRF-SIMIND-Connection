


               MPI-SIMIND Monte Carlo Simulation Program    V8.0  
------------------------------------------------------------------------------
 Phantom S : h2o       Crystal...: nai       InputFile.: simind            
 Phantom B : bone      BackScatt.: pmt       OutputFile: simind            
 Collimator: pb_sb2    SourceRout: smap      SourceImg.: source_ps         
 Cover.....: al        ScoreRout.: scattwin  DensityImg: density           
------------------------------------------------------------------------------
 PhotonEnergy.......: 150          y90_fre   PhotonsPerProj....: 655300000      
 EnergyResolution...: 9.4          Spectra   Activity..........: 1              
 MaxScatterOrder....: 10           ma-mleg   DetectorLenght....: 20.5           
 DetectorWidth......: 27.25        SPECT     DetectorHeight....: 0.95           
 UpperEneWindowTresh: 2230         x-rays    Distance to det...: 20             
 LowerEneWindowTresh: 30           BScatt    ShiftSource X.....: 0              
 PixelSize  I.......: 0.2          Random    ShiftSource Y.....: 0              
 PixelSize  J.......: 0.2          Cover     ShiftSource Z.....: 0              
 HalfLength S.......: 3.2          Phantom   HalfLength P......: 3.2            
 HalfWidth  S.......: 3.2          Resolut   HalfWidth  P......: 3.2            
 HalfHeight S.......: 3.2          Forced    HalfHeight P......: 3.2            
 SourceType.........: Integer2Map  Header    PhantomType.......: Integer2Map  
------------------------------------------------------------------------------
 GENERAL DATA
 keV/channel........: 5                      CutoffEnergy......: 10             
 Photons/Bq.........: 0.0334                 StartingAngle.....: 0              
 CameraOffset X.....: 0                      CoverThickness....: 0.1            
 CameraOffset Y.....: 0                      BackscatterThickn.: 1.6            
 MatrixSize I.......: 64                     IntrinsicResolut..: 0.32           
 MatrixSize J.......: 64                     AcceptanceAngle...: 89.9           
 Emission type......: -89.9                  Initial Weight....: 0.00005        
 NN ScalingFactor...: 100000                 Energy Channels...: 512            
                                                                              
 SPECT DATA
 RotationMode.......: -360                   Nr of Projections.: 1              
 RotationAngle......: 180                    Projection.[start]: 1              
 Orbital fraction...: 1                      Projection...[end]: 1              
                                                                              
 COLLIMATOR DATA FOR ROUTINE: MC RayTracing       
 CollimatorCode.....: ma-mleg                CollimatorType....: Parallel 
 HoleSize X.........: 0.19                   Distance X........: 0.05           
 HoleSize Y.........: 0.21939                Distance Y........: 0.153          
 CenterShift X......: 0.12                   X-Ray flag........: T              
 CenterShift Y......: 0.20785                CollimThickness...: 2.8            
 HoleShape..........: Hexagonal              Space Coll2Det....: 0              
 CollDepValue [57]..: 0                      CollDepValue [58].: 0              
 CollDepValue [59]..: 0                      CollDepValue [60].: 0              

 PHOTONS AFTER COLLIMATOR AND WITHIN ENER-WIN
 Geometric..........:   6.31 %          46.85 %
 Penetration........:  62.67 %          32.24 %
 Scatter in collim..:  29.71 %          18.19 %
 X-rays in collim...:   1.32 %           2.73 %

 K-alpha elem1 74.2 keV  63.56 %  69.13 %
 K-beta  elem1 85.4 keV  26.17 %  28.24 %
 K-alpha elem2 26.3 keV   8.21 %   0.76 %
 K-beta  elem2 29.8 keV   2.06 %   1.87 %
 K-alpha elem3  0.0 keV   0.00 %   0.00 %
 K-beta  elem3  0.0 keV   0.00 %   0.00 %
                                                                              
 IMAGE-BASED PHANTOM DATA
 RotationCentre.....:  33, 33                Bone definition...: 1170           
 CT-Pixel size......: 0.2                    Slice thickness...: 0.1            
 StartImage.........: 1                      No of CT-Images...: 64             
 MatrixSize I.......: 64                     CTmapOrientation..: 0              
 MatrixSize J.......: 64                     StepSize..........: 0.01           
 CenterPoint I......: 33                     ShiftPhantom X....: 0              
 CenterPoint J......: 33                     ShiftPhantom Y....: 0              
 CenterPoint K......: 33                     ShiftPhantom Z....: 0              
                                                                              
 INFO FOR TCT file
 MatrixSize I.......: 64                     MatrixSize J......: 64             
 MatrixSize K.......: 64                     Units.............: g/cm3*1000          
 Scout File.........: F
                                                                              
 Bremsstrahlung file: y90_frey.isd
 First energy used:     30.00keV
  Photon energy       Abundance
     30.000 keV       0.5601E-02
     40.000 keV       0.4023E-02
     50.000 keV       0.3271E-02
     60.000 keV       0.2621E-02
     70.000 keV       0.2080E-02
     80.000 keV       0.1727E-02
     90.000 keV       0.1364E-02
    100.000 keV       0.1159E-02
    110.000 keV       0.1019E-02
    120.000 keV       0.8852E-03
    130.000 keV       0.7762E-03
    140.000 keV       0.6834E-03
    150.000 keV       0.6038E-03
    160.000 keV       0.5390E-03
    170.000 keV       0.4818E-03
    180.000 keV       0.4313E-03
    190.000 keV       0.3889E-03
    200.000 keV       0.3522E-03
    210.000 keV       0.3193E-03
    220.000 keV       0.2907E-03
    230.000 keV       0.2657E-03
    240.000 keV       0.2425E-03
    250.000 keV       0.2236E-03
    260.000 keV       0.2066E-03
    270.000 keV       0.1910E-03
    280.000 keV       0.1767E-03
    290.000 keV       0.1649E-03
    300.000 keV       0.1536E-03
    310.000 keV       0.1437E-03
    320.000 keV       0.1350E-03
    330.000 keV       0.1264E-03
    340.000 keV       0.1187E-03
    350.000 keV       0.1113E-03
    360.000 keV       0.1049E-03
    370.000 keV       0.9908E-04
    380.000 keV       0.9420E-04
    390.000 keV       0.8883E-04
    400.000 keV       0.8438E-04
    410.000 keV       0.7989E-04
    420.000 keV       0.7620E-04
    430.000 keV       0.7237E-04
    440.000 keV       0.6918E-04
    450.000 keV       0.6603E-04
    460.000 keV       0.6254E-04
    470.000 keV       0.5983E-04
    480.000 keV       0.5718E-04
    490.000 keV       0.5485E-04
    500.000 keV       0.5252E-04
    510.000 keV       0.5028E-04
    520.000 keV       0.4787E-04
    530.000 keV       0.4593E-04
    540.000 keV       0.4394E-04
    550.000 keV       0.4240E-04
    560.000 keV       0.4067E-04
    570.000 keV       0.3906E-04
    580.000 keV       0.3761E-04
    590.000 keV       0.3615E-04
    600.000 keV       0.3493E-04
    610.000 keV       0.3344E-04
    620.000 keV       0.3237E-04
    630.000 keV       0.3104E-04
    640.000 keV       0.3025E-04
    650.000 keV       0.2902E-04
    660.000 keV       0.2822E-04
    670.000 keV       0.2703E-04
    680.000 keV       0.2599E-04
    690.000 keV       0.2529E-04
    700.000 keV       0.2448E-04
    710.000 keV       0.2337E-04
    720.000 keV       0.2293E-04
    730.000 keV       0.2195E-04
    740.000 keV       0.2123E-04
    750.000 keV       0.2073E-04
    760.000 keV       0.1997E-04
    770.000 keV       0.1941E-04
    780.000 keV       0.1873E-04
    790.000 keV       0.1807E-04
    800.000 keV       0.1763E-04
    810.000 keV       0.1702E-04
    820.000 keV       0.1652E-04
    830.000 keV       0.1597E-04
    840.000 keV       0.1553E-04
    850.000 keV       0.1494E-04
    860.000 keV       0.1448E-04
    870.000 keV       0.1404E-04
    880.000 keV       0.1361E-04
    890.000 keV       0.1327E-04
    900.000 keV       0.1285E-04
    910.000 keV       0.1242E-04
    920.000 keV       0.1211E-04
    930.000 keV       0.1173E-04
    940.000 keV       0.1150E-04
    950.000 keV       0.1097E-04
    960.000 keV       0.1067E-04
    970.000 keV       0.1049E-04
    980.000 keV       0.1007E-04
    990.000 keV       0.9755E-05
   1000.000 keV       0.9457E-05
   1010.000 keV       0.9201E-05
   1020.000 keV       0.8841E-05
   1030.000 keV       0.8653E-05
   1040.000 keV       0.8328E-05
   1050.000 keV       0.8081E-05
   1060.000 keV       0.7984E-05
   1070.000 keV       0.7688E-05
   1080.000 keV       0.7384E-05
   1090.000 keV       0.6943E-05
   1100.000 keV       0.6973E-05
   1110.000 keV       0.6714E-05
   1120.000 keV       0.6182E-05
   1130.000 keV       0.6448E-05
   1140.000 keV       0.5970E-05
   1150.000 keV       0.5779E-05
   1160.000 keV       0.5547E-05
   1170.000 keV       0.5380E-05
   1180.000 keV       0.5125E-05
   1190.000 keV       0.5054E-05
   1200.000 keV       0.4798E-05
   1210.000 keV       0.4615E-05
   1220.000 keV       0.4492E-05
   1230.000 keV       0.4224E-05
   1240.000 keV       0.4153E-05
   1250.000 keV       0.3850E-05
   1260.000 keV       0.3831E-05
   1270.000 keV       0.3592E-05
   1280.000 keV       0.3440E-05
   1290.000 keV       0.3362E-05
   1300.000 keV       0.3181E-05
   1310.000 keV       0.2967E-05
   1320.000 keV       0.2908E-05
   1330.000 keV       0.2739E-05
   1340.000 keV       0.2574E-05
   1350.000 keV       0.2539E-05
   1360.000 keV       0.2416E-05
   1370.000 keV       0.2234E-05
   1380.000 keV       0.2144E-05
   1390.000 keV       0.2076E-05
   1400.000 keV       0.1909E-05
   1410.000 keV       0.1839E-05
   1420.000 keV       0.1738E-05
   1430.000 keV       0.1639E-05
   1440.000 keV       0.1566E-05
   1450.000 keV       0.1476E-05
   1460.000 keV       0.1398E-05
   1470.000 keV       0.1321E-05
   1480.000 keV       0.1245E-05
   1490.000 keV       0.1187E-05
   1500.000 keV       0.1112E-05
   1510.000 keV       0.1036E-05
   1520.000 keV       0.9889E-06
   1530.000 keV       0.9283E-06
   1540.000 keV       0.8714E-06
   1550.000 keV       0.8341E-06
   1560.000 keV       0.7742E-06
   1570.000 keV       0.7260E-06
   1580.000 keV       0.6905E-06
   1590.000 keV       0.6464E-06
   1600.000 keV       0.6192E-06
   1610.000 keV       0.6007E-06
   1620.000 keV       0.5635E-06
   1630.000 keV       0.5410E-06
   1640.000 keV       0.5179E-06
   1650.000 keV       0.4838E-06
   1660.000 keV       0.4464E-06
   1670.000 keV       0.4057E-06
   1680.000 keV       0.3687E-06
   1690.000 keV       0.3552E-06
   1700.000 keV       0.3222E-06
   1710.000 keV       0.3284E-06
   1720.000 keV       0.3130E-06
   1730.000 keV       0.2902E-06
   1740.000 keV       0.2816E-06
   1750.000 keV       0.2704E-06
   1760.000 keV       0.2604E-06
   1770.000 keV       0.2533E-06
   1780.000 keV       0.2211E-06
   1790.000 keV       0.2113E-06
   1800.000 keV       0.2014E-06
   1810.000 keV       0.1906E-06
   1820.000 keV       0.1829E-06
   1830.000 keV       0.1752E-06
   1840.000 keV       0.1656E-06
   1850.000 keV       0.1528E-06
   1860.000 keV       0.1381E-06
   1870.000 keV       0.1384E-06
   1880.000 keV       0.1433E-06
   1890.000 keV       0.1378E-06
   1900.000 keV       0.1346E-06
   1910.000 keV       0.1258E-06
   1920.000 keV       0.1238E-06
   1930.000 keV       0.1293E-06
   1940.000 keV       0.1203E-06
   1950.000 keV       0.1140E-06
   1960.000 keV       0.1069E-06
   1970.000 keV       0.1014E-06
   1980.000 keV       0.1016E-06
   1990.000 keV       0.1066E-06
   2000.000 keV       0.1020E-06
------------------------------------------------------------------------------
  Scattwin results: Window file: simind.win          
  
  Win  WinAdded  Range(keV)   ScaleFactor
   1       0     36.8 - 126.2    1.00
   2       0     50.0 - 150.0    1.00
   3       0     75.0 - 225.0    1.00
   4       1     75.0 - 225.0    1.00
  
  Win    Total    Scatter   Primary  S/P-Ratio S/T Ratio  Cps/MBq
   1   0.455E+01 0.667E+00 0.389E+01 0.171E+00 0.146E+00 0.228E+01
   2   0.407E+01 0.538E+00 0.353E+01 0.152E+00 0.132E+00 0.204E+01
   3   0.372E+01 0.374E+00 0.335E+01 0.112E+00 0.100E+00 0.186E+01
   4   0.372E+01 0.374E+00 0.335E+01 0.112E+00 0.100E+00 0.186E+01
  
  Win  Geo(Air)  Pen(Air)  Sca(Air)  Geo(Tot)  Pen(Tot)  Sca(Tot)
   1    76.40%    12.58%    11.02%    76.81%    12.35%    10.84%
   2    72.40%    15.26%    12.35%    73.00%    14.90%    12.10%
   3    58.61%    25.44%    15.95%    59.36%    24.96%    15.68%
   4    58.61%    25.44%    15.95%    59.36%    24.96%    15.68%
  
  Win   SC 1  SC 2  SC 3  SC 4  SC 5  SC 6  SC 7  SC 8  SC 9  SC10
   1   85.1% 12.1%  2.2%  0.5%  0.1%  0.0%  0.0%  0.0%  0.0%  0.0%
   2   87.2% 11.0%  1.5%  0.2%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
   3   90.7%  8.5%  0.7%  0.1%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
   4   90.7%  8.5%  0.7%  0.1%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
                                                                              
 INTERACTIONS IN THE CRYSTAL
 MaxValue spectrum..: 0.8135         
 MaxValue projection: 0.6806E-01     
 CountRate spectrum.: 20.69          
 CountRate E-Window.: 5.354          
                                                                              
 SCATTER IN ENERGY WINDOW
 Scatter/Primary....: 0.11063        
 Scatter/Total......: 0.09961        
 Scatter order 1....: 87.08 %        
 Scatter order 2....: 10.09 %        
 Scatter order 3....: 2.04 %         
 Scatter order 4....: 0.54 %         
 Scatter order 5....: 0.16 %         
 Scatter order 6....: 0.05 %         
 Scatter order 7....: 0.02 %         
 Scatter order 8....: 0.01 %         
 Scatter order 9....: 0    %         
 Scatter order10....: 0    %         
                                                                              
 CALCULATED DETECTOR PARAMETERS
 Efficiency E-window: 0.0858         
 Efficiency spectrum: 0.3314         
 Sensitivity Cps/MBq: 5.354          
 Sensitivity Cpm/uCi: 11.8859        
                                                                              
 Simulation started.: 2025:10:02 15:11:58
 Simulation stopped.: 2025:10:02 15:37:49
 Elapsed time.......: 0 h, 25 m and 51 s
 DetectorHits.......: 17679004       
 DetectorHits/CPUsec: 11525          
                                                                              
 OTHER INFORMATION
 Compiled 2024:05:03 with INTEL Mac   
 Random number generator: Intel RAN
 MPI Processors: 2              
 Comment:EMISSION
 Energy resolution as function of 1/sqrt(E)
 Header file: simind.h00
 Linear angle sampling within acceptance angle
 Inifile: simind.ini
 Command: simind.smc /MP/CC:ma-mlegp/PX:0.2/TH:0.2/26:10/NN:100000/CA:1/FI:y90_frey/12:20/53:1
