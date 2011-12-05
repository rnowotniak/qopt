// test_project.cpp : 定義主控台應用程式的進入點。
//
#include <iostream>
#include "mCEC_Function.h"
int main()
{
  FIELD_TYPE x[1000];
  FIELD_TYPE objective_value[10];
  int i;
  std::cout<<"Initial cost function"<<std::endl;
  //Initial Cost Function
  if(  Initial_CEC2011_Cost_Function() == _NO_ERROR)
  {
    std::cout<< "Initial cost function success" <<std::endl;
    x[0]=1;x[1]=5;x[2]=-1.5;x[3]=4.8;x[4]=2;x[5]=4.9;
    std::cout<<"Run Cost Function 1"<<std::endl;
    cost_function1(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    x[0]=0;x[1]=0.1;x[2]=0.2;x[3]=0.3;x[4]=0.4;x[5]=0.5;x[6]=0.6;x[7]=0.7;x[8]=0.8;x[9]=0.9;
    x[10]=1.0;x[11]=1.1;x[12]=1.2;x[13]=1.3;x[14]=1.4;x[15]=1.5;x[16]=1.6;x[17]=1.7;x[18]=1.8;x[19]=1.9;
    x[20]=0;x[21]=0;x[22]=0;x[23]=0;x[24]=0;x[25]=0;x[26]=0;x[27]=0;x[28]=0;x[29]=0;
    x[0]=0;x[1]=0;x[2]=0;x[3]=0;x[4]=0;x[5]=0;x[6]=0;x[7]=0;x[8]=0;x[9]=0;
    x[10]=0;x[11]=0;x[12]=0;x[13]=0;x[14]=0;x[15]=0;x[16]=0;x[17]=0;x[18]=0;x[19]=0;
    x[20]=0;x[21]=0;x[22]=0;x[23]=0;x[24]=0;x[25]=0;x[26]=0;x[27]=0;x[28]=0;x[29]=0;
    std::cout<<"Run Cost Function 2"<<std::endl;
    cost_function2(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    x[0] = 0.7;
    std::cout<<"Run Cost Function 3"<<std::endl;
    cost_function3(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    x[0] = 0.1;
    std::cout<<"Run Cost Function 4"<<std::endl;
    cost_function4(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 5"<<std::endl;
    x[0]=0;x[1]=0;x[2]=0;x[3]=0;x[4]=0;x[5]=0;x[6]=0;x[7]=0;x[8]=0;x[9]=0;
    x[10]=0;x[11]=0;x[12]=0;x[13]=0;x[14]=0;x[15]=0;x[16]=0;x[17]=0;x[18]=0;x[19]=0;
    x[20]=0;x[21]=0;x[22]=0;x[23]=0;x[24]=0;x[25]=0;x[26]=0;x[27]=0;x[28]=0;x[29]=0;
    cost_function5(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 6"<<std::endl;
    cost_function6(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 7"<<std::endl;
    cost_function7(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 8"<<std::endl;
    x[0]=1;x[1]=1;x[2]=1;x[3]=1;x[4]=1;x[5]=1;x[6]=1;
    cost_function8(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 9"<<std::endl;
    cost_function9(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 10"<<std::endl;
    x[0]=0.3;x[1]=0.3;x[2]=0.3;x[3]=0.3;x[4]=0.3;x[5]=0.3;x[6]=0.3;x[7]=0.3;x[8]=0.3;x[9]=0.3;
    x[10]=0;x[11]=0;x[12]=0;
    cost_function10(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    for(i=0;i<120;)
    {
      x[i]=10;
      i++;
      x[i]=20;
      i++;
      x[i]=30;
      i++;
      x[i]=40;
      i++;
      x[i]=50;
      i++;
    }
    std::cout<<"Run Cost Function 11 with dimension 120"<<std::endl;
    cost_function11_5(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    for(i=0;i<240;)
    {
      x[i++]=150;
      x[i++]=135;
      x[i++]=73;
      x[i++]=60;
      x[i++]=73;
      x[i++]=57;
      x[i++]=20;
      x[i++]=47;
      x[i++]=20;
      x[i++]=55;
    }
    std::cout<<"Run Cost Function 11 with dimension 240"<<std::endl;
    cost_function11_10(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;

    std::cout<<"Run Cost Function 12_6"<<std::endl;
    x[0]=100;x[1]=50;x[2]=80;x[3]=50;x[4]=50;x[5]=50;
    cost_function12_6(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 12_13"<<std::endl;
    x[0]=0;x[1]=0;x[2]=0;x[3]=60;x[4]=60;x[5]=60;x[6]=60;x[7]=60;x[8]=60;x[9]=40;
    x[10]=40;x[11]=55;x[12]=55;
    cost_function12_13(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 12_15"<<std::endl;
    x[0]=150;x[1]=150;x[2]=20;x[3]=20;x[4]=150;x[5]=135;x[6]=135;x[7]=60;x[8]=25;x[9]=25;
    x[10]=20;x[11]=20;x[12]=25;x[13]=15;x[14]=15;
    cost_function12_15(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 12_40"<<std::endl;
    x[0]=36;x[1]=36;x[2]=60;x[3]=80;x[4]=47;x[5]=68;x[6]=110;x[7]=135;x[8]=135;x[9]=130;
    x[10]=94;x[11]=94;x[12]=125;x[13]=125;x[14]=125;x[15]=125;x[16]=220;x[17]=220;x[18]=242;x[19]=242;
    x[20]=254;x[21]=254;x[22]=254;x[23]=254;x[24]=254;x[25]=254;x[26]=10;x[27]=10;x[28]=10;x[29]=47;
    x[30]=60;x[31]=60;x[32]=60;x[33]=90;x[34]=90;x[35]=90;x[36]=25;x[37]=25;x[38]=25;x[39]=242;
    cost_function12_40(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 12_140"<<std::endl;
    x[0]=71;x[1]=120;x[2]=125;x[3]=125;x[4]=90;x[5]=90;x[6]=280;x[7]=280;x[8]=260;x[9]=260;
    x[10]=260;x[11]=260;x[12]=260;x[13]=260;x[14]=260;x[15]=260;x[16]=260;x[17]=260;x[18]=260;x[19]=260;
    x[20]=260;x[21]=60;x[22]=260;x[23]=260;x[24]=280;x[25]=280;x[26]=280;x[27]=280;x[28]=260;x[29]=260;
    x[30]=260;x[31]=260;x[32]=260;x[33]=260;x[34]=260;x[35]=260;x[36]=120;x[37]=120;x[38]=423;x[39]=423;
    x[40]=3;x[41]=3;x[42]=160;x[43]=160;x[44]=160;x[45]=160;x[46]=160;x[47]=160;x[48]=160;x[49]=160;
    x[50]=165;x[51]=165;x[52]=165;x[53]=165;x[54]=180;x[55]=180;x[56]=103;x[57]=198;x[58]=100;x[59]=153;
    x[60]=163;x[61]=95;x[62]=160;x[63]=160;x[64]=196;x[65]=196;x[66]=196;x[67]=196;x[68]=130;x[69]=130;
    x[70]=137;x[71]=137;x[72]=195;x[73]=175;x[74]=175;x[75]=175;x[76]=175;x[77]=330;x[78]=160;x[79]=160;
    x[80]=200;x[81]=56;x[82]=115;x[83]=115;x[84]=115;x[85]=207;x[86]=207;x[87]=175;x[88]=175;x[89]=175;
    x[90]=175;x[91]=360;x[92]=415;x[93]=795;x[94]=795;x[95]=578;x[96]=615;x[97]=612;x[98]=612;x[99]=758;
    x[100]=755;x[101]=750;x[102]=750;x[103]=713;x[104]=718;x[105]=791;x[106]=786;x[107]=795;x[108]=795;x[109]=795;
    x[110]=795;x[111]=94;x[112]=94;x[113]=94;x[114]=244;x[115]=244;x[116]=244;x[117]=95;x[118]=95;x[119]=116;
    x[120]=175;x[121]=2;x[122]=4;x[123]=15;x[124]=9;x[125]=12;x[126]=10;x[127]=112;x[128]=4;x[129]=5;
    x[130]=5;x[131]=50;x[132]=5;x[133]=42;x[134]=42;x[135]=41;x[136]=17;x[137]=7;x[138]=7;x[139]=26;
    cost_function12_140(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 13_1"<<std::endl;
    for(i=0;i<96;)
    {
      x[i++]=5;
      x[i++]=6;
      x[i++]=10;
      x[i++]=13;
    }
    cost_function13_1(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 13_2"<<std::endl;
    cost_function13_2(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 13_3"<<std::endl;
    cost_function13_3(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;
    std::cout<<"Run Cost Function 14"<<std::endl;
    x[0]=1900;x[1]=2.5;x[2]=0;x[3]=0;x[4]=100;x[5]=100;x[6]=100;x[7]=100;x[8]=100;x[9]=100;
    x[10]=0.01;x[11]=0.01;x[12]=0.01;x[13]=0.01;x[14]=0.01;x[15]=0.01;x[16]=1.1;x[17]=1.1;x[18]=1.05;x[19]=1.05;
    x[20]=1.05;x[21]=0;x[22]=0;x[23]=0;x[24]=0;x[25]=0;
    cost_function14(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;

    std::cout<<"Run Cost Function 15"<<std::endl;
    x[0]=-1000;x[1]=3;x[2]=0;x[3]=0;x[4]=100;x[5]=100;x[6]=30;x[7]=400;x[8]=800;x[9]=0.01;
    x[10]=0.01;x[11]=0.01;x[12]=0.01;x[13]=0.01;x[14]=1.05;x[15]=1.05;x[16]=1.15;x[17]=1.7;x[18]=0;x[19]=0;
    x[20]=0;x[21]=0;
    cost_function15(x, objective_value);
    std::cout<<"objective="<<objective_value[0]<<std::endl;



  }
  // Terminate Cost Function
  Terminate_CEC2011_Cost_Function();
  std::cin.get();
  return 0;
}
