OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
u3(0,0.07485071764072027,0.07485071764072027) q[1];
u3(0,0,0.19516810404566803) q[2];
u3(0,0,0.19516810404566806) q[3];
u3(0,0,0.27278629553012745) q[4];
u3(0,0,0.2727862955301274) q[5];
cx q[5],q[4];
u3(0,0,0.08091332405325455) q[4];
cx q[5],q[3];
u3(0,0,0.048697964824652606) q[3];
cx q[4],q[3];
u3(0,0,0.06444430414237483) q[3];
cx q[5],q[2];
u3(0,0,0.06444430414237483) q[2];
cx q[4],q[2];
u3(0,0,0.048697964824652606) q[2];
cx q[3],q[2];
u3(0,0,0.06373465784388439) q[2];
cx q[5],q[1];
cx q[0],q[5];
u3(pi/2,0,-2.991891218308353) q[0];
u3(0,0,0.06339688831389195) q[5];
u3(0,0.02641727029059382,0.02641727029059382) q[1];
cx q[4],q[1];
u3(pi/2,0,-3.0781957652759013) q[1];
cx q[5],q[4];
cx q[4],q[1];
u3(pi/2,0,-pi) q[1];
cx q[3],q[1];
u3(pi/2,0,-3.0966886735029537) q[1];
u3(0.011922473641009172,pi/2,-1.517961786213709) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[4],q[1];
u3(pi/2,0,pi) q[1];
cx q[2],q[1];
u3(0,0,0.05945291055829548) q[1];
u3(0.011922473641009172,0,1.6302492373531923) q[4];
cx q[2],q[4];
u3(pi/2,0,pi) q[2];
cx q[4],q[3];
u3(0.01192247364100917,-pi,-3.0966886735029537) q[4];
cx q[5],q[1];
u3(pi/2,0,pi) q[1];
u3(0.0012622097443250315,-pi/2,pi/2) q[5];
cx q[5],q[1];
u3(0.00391847450165415,pi/2,-pi/2) q[5];
cx q[5],q[0];
u3(0.024660238369096584,-pi,-pi/2) q[5];
cx q[4],q[5];
cx q[5],q[0];
u3(1.5720585365392217,-pi/2,-pi) q[5];
cx q[5],q[1];
u3(0.059737170520974665,3.0759113293488802,-3.0760281516138708) q[5];
cx q[5],q[0];
cx q[0],q[4];
cx q[1],q[0];
cx q[0],q[3];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,3.129670179948784) q[3];
cx q[3],q[1];
u3(pi/2,-0.015746339317722402,-pi) q[1];
u3(0.013184683385334201,-pi/2,pi/2) q[3];
cx q[3],q[2];
u3(0.024660238369096584,-pi,-pi/2) q[5];
cx q[5],q[4];
u3(pi/2,pi/2,-pi) q[4];
u3(pi/2,0,-pi) q[5];
cx q[5],q[2];
u3(pi/2,0,pi) q[2];
u3(0.011922473641009169,0,0) q[5];
cx q[5],q[3];
u3(0.0012622097443250315,-1.586542666112619,pi/2) q[3];
cx q[5],q[2];
u3(0,0,1.5865426661126187) q[2];
u3(pi/2,0,-pi) q[5];
cx q[5],q[3];
u3(0,0,pi/2) q[3];
cx q[4],q[3];
u3(pi/2,0.01192247364100929,-pi) q[5];
cx q[5],q[2];
u3(0.014548930471455985,-pi,-pi/2) q[5];
cx q[0],q[5];
u3(pi/2,0,pi) q[0];
cx q[3],q[0];
u3(pi/2,0,pi) q[0];
cx q[0],q[4];
cx q[1],q[0];
u3(0.013184683385334201,pi/2,-pi/2) q[1];
u3(0.024660238369096588,-pi,-pi) q[3];
cx q[2],q[3];
u3(0.014548930471455987,pi/2,0) q[2];
u3(0,pi/4,pi/4) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,-0.024660238369096366,-pi) q[1];
cx q[4],q[1];
u3(0,0,0.0012622097443250315) q[1];
cx q[4],q[3];
u3(0,0,-pi/2) q[4];
u3(0.0105623477327041,-0.003918474501654501,0) q[5];
cx q[1],q[5];
cx q[0],q[1];
u3(0.014548930471455985,-pi,-pi) q[1];
u3(1.5576116434095624,-2.4551356701246734,-pi) q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[0],q[1];
u3(pi/2,0,-pi) q[1];
u3(0.01192247364100917,pi/2,0) q[2];
u3(0.01880975082179262,0.884252611152542,pi/2) q[5];
cx q[5],q[2];
u3(0.013184683385334201,-pi/2,pi/2) q[5];
cx q[5],q[1];
u3(pi/2,0,-pi) q[1];
u3(0.01764268771791049,0.9289176615991668,0.64180403059955) q[5];
cx q[1],q[5];
u3(pi/2,0.014131831698376729,-pi) q[1];
cx q[2],q[1];
u3(pi/2,0,pi) q[2];
u3(0,0.0070659158491883645,0.0070659158491883645) q[1];
cx q[3],q[2];
u3(0,0,0.01574633931772222) q[2];
cx q[3],q[0];
u3(pi/2,0,pi) q[0];
cx q[5],q[0];
u3(pi/2,0,pi) q[0];
cx q[3],q[0];
u3(pi/2,-0.003918474501654057,-pi) q[5];
cx q[5],q[1];
u3(0,-pi/2,-pi/2) q[1];
u3(1.5602339790621924,0.6418040305995505,-pi) q[5];
cx q[5],q[2];
cx q[4],q[2];
cx q[4],q[0];
u3(pi/2,0,pi) q[0];
cx q[2],q[0];
cx q[3],q[0];
u3(pi/2,0,-pi) q[0];
u3(0.017642687717910492,-2.212674991990626,pi/2) q[5];
cx q[5],q[1];
u3(pi/2,0,pi/2) q[1];
cx q[3],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[3];
cx q[4],q[1];
u3(pi/2,0,pi) q[4];
u3(pi/2,pi/2,-pi) q[1];
cx q[1],q[0];
u3(pi/2,0,-pi) q[0];
cx q[5],q[4];
u3(pi/2,0,pi) q[5];
cx q[5],q[2];
cx q[3],q[2];
u3(pi/2,0,-pi) q[2];
cx q[5],q[4];
cx q[4],q[3];
u3(pi/2,0,pi) q[3];
cx q[4],q[0];
u3(pi/2,pi/2,-pi) q[0];
u3(pi/2,-pi,-pi) q[4];
cx q[5],q[1];
u3(0,0,-pi) q[1];
cx q[2],q[1];
u3(0,pi/4,pi/4) q[1];
cx q[0],q[1];
u3(pi,-pi/2,-pi/2) q[1];
u3(pi,-pi/2,-pi/2) q[2];
