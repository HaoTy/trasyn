OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
cx q[0],q[13];
rz(5.354653508904871) q[13];
cx q[0],q[13];
cx q[0],q[15];
rz(4.471358549692117) q[15];
cx q[0],q[15];
cx q[16],q[0];
rz(4.471828808588433) q[0];
cx q[16],q[0];
cx q[1],q[3];
rz(5.688115766004765) q[3];
cx q[1],q[3];
cx q[1],q[12];
rz(4.902819178044501) q[12];
cx q[1],q[12];
cx q[19],q[1];
rz(4.9229377722707595) q[1];
cx q[19],q[1];
cx q[2],q[4];
rz(5.034951729317795) q[4];
cx q[2],q[4];
cx q[2],q[11];
rz(4.913206542091023) q[11];
cx q[2],q[11];
cx q[22],q[2];
rz(5.100270784455824) q[2];
cx q[22],q[2];
cx q[3],q[10];
rz(6.335278578795391) q[10];
cx q[3],q[10];
cx q[19],q[3];
rz(5.950328858650351) q[3];
cx q[19],q[3];
cx q[4],q[7];
rz(4.738352064883333) q[7];
cx q[4],q[7];
cx q[23],q[4];
rz(5.443931714593741) q[4];
cx q[23],q[4];
cx q[5],q[8];
rz(6.012932643389077) q[8];
cx q[5],q[8];
cx q[5],q[18];
rz(5.4374503667353355) q[18];
cx q[5],q[18];
cx q[21],q[5];
rz(5.831014920663796) q[5];
cx q[21],q[5];
cx q[6],q[7];
rz(4.661488008876823) q[7];
cx q[6],q[7];
cx q[6],q[14];
rz(5.526429849234434) q[14];
cx q[6],q[14];
cx q[15],q[6];
rz(5.320219624045379) q[6];
cx q[15],q[6];
cx q[10],q[7];
rz(5.574799040426) q[7];
cx q[10],q[7];
cx q[8],q[18];
rz(5.3919057141260875) q[18];
cx q[8],q[18];
cx q[20],q[8];
rz(4.708398621368499) q[8];
cx q[20],q[8];
cx q[9],q[11];
rz(4.799518523670533) q[11];
cx q[9],q[11];
cx q[9],q[12];
rz(4.525106801286123) q[12];
cx q[9],q[12];
cx q[14],q[9];
rz(5.67922108561664) q[9];
cx q[14],q[9];
cx q[16],q[10];
rz(4.926551182399837) q[10];
cx q[16],q[10];
cx q[17],q[11];
rz(5.64852342092286) q[11];
cx q[17],q[11];
cx q[20],q[12];
rz(4.742214685443978) q[12];
cx q[20],q[12];
cx q[13],q[22];
rz(5.389329437809927) q[22];
cx q[13],q[22];
cx q[23],q[13];
rz(5.043481352758028) q[13];
cx q[23],q[13];
cx q[23],q[14];
rz(5.14936462423533) q[14];
cx q[23],q[14];
cx q[18],q[15];
rz(5.368647791221526) q[15];
cx q[18],q[15];
cx q[17],q[16];
rz(5.035645770118487) q[16];
cx q[17],q[16];
cx q[19],q[17];
rz(5.02491028159904) q[17];
cx q[19],q[17];
cx q[21],q[20];
rz(4.628740951545873) q[20];
cx q[21],q[20];
cx q[22],q[21];
rz(5.712688203923747) q[21];
cx q[22],q[21];
rx(3.456421153032202) q[0];
rx(3.456421153032202) q[1];
rx(3.456421153032202) q[2];
rx(3.456421153032202) q[3];
rx(3.456421153032202) q[4];
rx(3.456421153032202) q[5];
rx(3.456421153032202) q[6];
rx(3.456421153032202) q[7];
rx(3.456421153032202) q[8];
rx(3.456421153032202) q[9];
rx(3.456421153032202) q[10];
rx(3.456421153032202) q[11];
rx(3.456421153032202) q[12];
rx(3.456421153032202) q[13];
rx(3.456421153032202) q[14];
rx(3.456421153032202) q[15];
rx(3.456421153032202) q[16];
rx(3.456421153032202) q[17];
rx(3.456421153032202) q[18];
rx(3.456421153032202) q[19];
rx(3.456421153032202) q[20];
rx(3.456421153032202) q[21];
rx(3.456421153032202) q[22];
rx(3.456421153032202) q[23];
cx q[0],q[13];
rz(4.719732660535021) q[13];
cx q[0],q[13];
cx q[0],q[15];
rz(3.9411732148212293) q[15];
cx q[0],q[15];
cx q[16],q[0];
rz(3.941587713400486) q[0];
cx q[16],q[0];
cx q[1],q[3];
rz(5.0136550783483775) q[3];
cx q[1],q[3];
cx q[1],q[12];
rz(4.321473978630349) q[12];
cx q[1],q[12];
cx q[19],q[1];
rz(4.339207037566061) q[1];
cx q[19],q[1];
cx q[2],q[4];
rz(4.437939090094104) q[4];
cx q[2],q[4];
cx q[2],q[11];
rz(4.330629674935575) q[11];
cx q[2],q[11];
cx q[22],q[2];
rz(4.495513025994447) q[2];
cx q[22],q[2];
cx q[3],q[10];
rz(5.584081429770007) q[10];
cx q[3],q[10];
cx q[19],q[3];
rz(5.244776605692987) q[3];
cx q[19],q[3];
cx q[4],q[7];
rz(4.176508332528361) q[7];
cx q[4],q[7];
cx q[23],q[4];
rz(4.798424822887445) q[4];
cx q[23],q[4];
cx q[5],q[8];
rz(5.299957230734943) q[8];
cx q[5],q[8];
cx q[5],q[18];
rz(4.79271199214672) q[18];
cx q[5],q[18];
cx q[21],q[5];
rz(5.139610157661252) q[5];
cx q[21],q[5];
cx q[6],q[7];
rz(4.108758328732259) q[7];
cx q[6],q[7];
cx q[6],q[14];
rz(4.871140852010407) q[14];
cx q[6],q[14];
cx q[15],q[6];
rz(4.689381727327078) q[6];
cx q[15],q[6];
cx q[10],q[7];
rz(4.913774731317602) q[7];
cx q[10],q[7];
cx q[8],q[18];
rz(4.752567735552879) q[18];
cx q[8],q[18];
cx q[20],q[8];
rz(4.150106578349991) q[8];
cx q[20],q[8];
cx q[9],q[11];
rz(4.230422060613124) q[11];
cx q[9],q[11];
cx q[9],q[12];
rz(3.9885483396678727) q[12];
cx q[9],q[12];
cx q[14],q[9];
rz(5.005815072741472) q[9];
cx q[14],q[9];
cx q[16],q[10];
rz(4.342391992441994) q[10];
cx q[16],q[10];
cx q[17],q[11];
rz(4.9787573424109395) q[11];
cx q[17],q[11];
cx q[20],q[12];
rz(4.179912948043635) q[12];
cx q[20],q[12];
cx q[13],q[22];
rz(4.750296937740897) q[22];
cx q[13],q[22];
cx q[23],q[13];
rz(4.445457324890432) q[13];
cx q[23],q[13];
cx q[23],q[14];
rz(4.538785629656491) q[14];
cx q[23],q[14];
cx q[18],q[15];
rz(4.732067589620689) q[15];
cx q[18],q[15];
cx q[17],q[16];
rz(4.43855083593897) q[16];
cx q[17],q[16];
cx q[19],q[17];
rz(4.429088293552657) q[17];
cx q[19],q[17];
cx q[21],q[20];
rz(4.079894209744118) q[20];
cx q[21],q[20];
cx q[22],q[21];
rz(5.035313872442584) q[21];
cx q[22],q[21];
rx(1.2555202816903654) q[0];
rx(1.2555202816903654) q[1];
rx(1.2555202816903654) q[2];
rx(1.2555202816903654) q[3];
rx(1.2555202816903654) q[4];
rx(1.2555202816903654) q[5];
rx(1.2555202816903654) q[6];
rx(1.2555202816903654) q[7];
rx(1.2555202816903654) q[8];
rx(1.2555202816903654) q[9];
rx(1.2555202816903654) q[10];
rx(1.2555202816903654) q[11];
rx(1.2555202816903654) q[12];
rx(1.2555202816903654) q[13];
rx(1.2555202816903654) q[14];
rx(1.2555202816903654) q[15];
rx(1.2555202816903654) q[16];
rx(1.2555202816903654) q[17];
rx(1.2555202816903654) q[18];
rx(1.2555202816903654) q[19];
rx(1.2555202816903654) q[20];
rx(1.2555202816903654) q[21];
rx(1.2555202816903654) q[22];
rx(1.2555202816903654) q[23];
