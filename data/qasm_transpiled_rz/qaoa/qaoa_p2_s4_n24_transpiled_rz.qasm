OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
h q[0];
h q[1];
h q[2];
cx q[0],q[2];
rz(5.612986328646171) q[2];
cx q[0],q[2];
h q[3];
h q[4];
h q[5];
cx q[2],q[5];
rz(5.70338209274183) q[5];
cx q[2],q[5];
h q[6];
cx q[4],q[6];
rz(4.363091356766081) q[6];
cx q[4],q[6];
h q[7];
h q[8];
h q[9];
cx q[6],q[9];
rz(5.767481478429907) q[9];
cx q[6],q[9];
h q[10];
cx q[1],q[10];
rz(5.516651273618305) q[10];
cx q[1],q[10];
cx q[7],q[10];
rz(5.537639349750145) q[10];
cx q[7],q[10];
h q[11];
cx q[9],q[11];
rz(4.997510833266734) q[11];
cx q[9],q[11];
cx q[11],q[10];
rz(-3.0533840388531446) q[10];
h q[10];
rz(0.8202907910159238) q[10];
h q[10];
rz(3*pi) q[10];
cx q[11],q[10];
h q[12];
cx q[12],q[6];
rz(-3.152024285633068) q[6];
h q[6];
rz(0.8202907910159243) q[6];
h q[6];
rz(3*pi) q[6];
cx q[12],q[6];
h q[13];
cx q[13],q[2];
rz(-4.038542836292723) q[2];
h q[2];
rz(0.8202907910159243) q[2];
h q[2];
rz(3*pi) q[2];
cx q[13],q[2];
cx q[5],q[13];
rz(5.666177128500306) q[13];
cx q[5],q[13];
cx q[13],q[11];
rz(-3.1747856765149414) q[11];
h q[11];
rz(0.8202907910159238) q[11];
h q[11];
rz(3*pi) q[11];
cx q[13],q[11];
h q[13];
rz(5.4628945161636615) q[13];
h q[13];
h q[14];
cx q[3],q[14];
rz(5.148038066228992) q[14];
cx q[3],q[14];
cx q[8],q[14];
rz(5.025806076040759) q[14];
cx q[8],q[14];
h q[15];
cx q[3],q[15];
rz(6.312907165649878) q[15];
cx q[3],q[15];
cx q[15],q[9];
rz(-3.5970210917033056) q[9];
h q[9];
rz(0.8202907910159238) q[9];
h q[9];
rz(3*pi) q[9];
cx q[15],q[9];
h q[16];
cx q[8],q[16];
rz(4.953380567868653) q[16];
cx q[8],q[16];
cx q[12],q[16];
rz(5.458378606350126) q[16];
cx q[12],q[16];
h q[17];
cx q[17],q[15];
rz(-3.8940962139380617) q[15];
h q[15];
rz(0.8202907910159243) q[15];
h q[15];
rz(3*pi) q[15];
cx q[17],q[15];
h q[18];
cx q[7],q[18];
rz(6.478034797901719) q[18];
cx q[7],q[18];
cx q[18],q[14];
rz(-3.966773553204837) q[14];
h q[14];
rz(0.8202907910159238) q[14];
h q[14];
rz(3*pi) q[14];
cx q[18],q[14];
h q[19];
cx q[0],q[19];
rz(5.126009548762815) q[19];
cx q[0],q[19];
cx q[1],q[19];
rz(6.374090220995531) q[19];
cx q[1],q[19];
cx q[19],q[16];
rz(-3.103647853966962) q[16];
h q[16];
rz(0.8202907910159238) q[16];
h q[16];
rz(3*pi) q[16];
cx q[19],q[16];
h q[20];
cx q[20],q[1];
rz(-3.24848223897803) q[1];
h q[1];
rz(0.8202907910159238) q[1];
h q[1];
rz(3*pi) q[1];
cx q[20],q[1];
cx q[1],q[10];
rz(7.955580847927095) q[10];
cx q[1],q[10];
cx q[20],q[12];
rz(-4.35538908913984) q[12];
h q[12];
rz(0.8202907910159238) q[12];
h q[12];
rz(3*pi) q[12];
cx q[20],q[12];
cx q[17],q[20];
rz(-4.01906676020955) q[20];
h q[20];
rz(0.8202907910159238) q[20];
h q[20];
rz(3*pi) q[20];
cx q[17],q[20];
h q[21];
cx q[21],q[3];
rz(-3.6654401591271397) q[3];
h q[3];
rz(0.8202907910159238) q[3];
h q[3];
rz(3*pi) q[3];
cx q[21],q[3];
cx q[3],q[14];
rz(7.843834236854072) q[14];
cx q[3],q[14];
cx q[3],q[15];
rz(8.196969099074266) q[15];
cx q[3],q[15];
cx q[4],q[21];
rz(6.891035026531464) q[21];
cx q[4],q[21];
cx q[21],q[5];
rz(-3.0458003288770112) q[5];
h q[5];
rz(0.8202907910159238) q[5];
h q[5];
rz(3*pi) q[5];
cx q[21],q[5];
h q[21];
rz(5.4628945161636615) q[21];
h q[21];
cx q[21],q[3];
rz(-4.537218346536886) q[3];
h q[3];
rz(1.5723751620236852) q[3];
h q[3];
cx q[21],q[3];
h q[22];
cx q[22],q[4];
rz(-3.0428912856904957) q[4];
h q[4];
rz(0.8202907910159238) q[4];
h q[4];
rz(3*pi) q[4];
cx q[22],q[4];
cx q[22],q[8];
rz(-4.173999066392315) q[8];
h q[8];
rz(0.8202907910159238) q[8];
h q[8];
rz(3*pi) q[8];
cx q[22],q[8];
cx q[22],q[17];
rz(-3.857935956656347) q[17];
h q[17];
rz(0.8202907910159238) q[17];
h q[17];
rz(3*pi) q[17];
cx q[22],q[17];
h q[22];
rz(5.4628945161636615) q[22];
h q[22];
cx q[4],q[6];
rz(7.6058744109524445) q[6];
cx q[4],q[6];
cx q[4],q[21];
rz(2.0890456325594964) q[21];
cx q[4],q[21];
cx q[22],q[4];
rz(-4.348490093074173) q[4];
h q[4];
rz(1.5723751620236852) q[4];
h q[4];
cx q[22],q[4];
cx q[6],q[9];
rz(8.031621056960748) q[9];
cx q[6],q[9];
cx q[12],q[6];
rz(-4.381574211210328) q[6];
h q[6];
rz(1.5723751620236852) q[6];
h q[6];
cx q[12],q[6];
cx q[8],q[14];
rz(1.5235937987284571) q[14];
cx q[8],q[14];
cx q[8],q[16];
rz(7.784823014994224) q[16];
cx q[8],q[16];
cx q[12],q[16];
rz(1.6547299418083807) q[16];
cx q[12],q[16];
cx q[22],q[8];
rz(-4.69139007255934) q[8];
h q[8];
rz(1.5723751620236852) q[8];
h q[8];
cx q[22],q[8];
cx q[9],q[11];
rz(7.7982012865947326) q[11];
cx q[9],q[11];
cx q[15],q[9];
rz(-4.51647682462686) q[9];
h q[9];
rz(1.5723751620236852) q[9];
h q[9];
cx q[15],q[9];
cx q[17],q[15];
rz(-4.606536370726565) q[15];
h q[15];
rz(1.5723751620236852) q[15];
h q[15];
cx q[17],q[15];
h q[23];
cx q[23],q[0];
rz(-3.364060779439012) q[0];
h q[0];
rz(0.8202907910159243) q[0];
h q[0];
rz(3*pi) q[0];
cx q[23],q[0];
cx q[0],q[2];
rz(7.984785216384591) q[2];
cx q[0],q[2];
cx q[0],q[19];
rz(-pi) q[19];
h q[19];
rz(0.8202907910159238) q[19];
h q[19];
rz(10.978748854702165) q[19];
cx q[0],q[19];
cx q[1],q[19];
rz(1.9323316869589722) q[19];
cx q[1],q[19];
cx q[19],q[16];
rz(-4.366908696827975) q[16];
h q[16];
rz(1.5723751620236852) q[16];
h q[16];
cx q[19],q[16];
h q[19];
rz(1.5723751620236852) q[19];
h q[19];
cx q[2],q[5];
rz(8.012189064349208) q[5];
cx q[2],q[5];
cx q[13],q[2];
rz(-4.650325958851742) q[2];
h q[2];
rz(1.5723751620236852) q[2];
h q[2];
cx q[13],q[2];
cx q[20],q[1];
rz(-4.410815836798515) q[1];
h q[1];
rz(1.5723751620236852) q[1];
h q[1];
cx q[20],q[1];
cx q[20],q[12];
rz(1.5368061026028297) q[12];
h q[12];
rz(1.5723751620236852) q[12];
h q[12];
cx q[20],q[12];
cx q[17],q[20];
rz(-4.644421706221433) q[20];
h q[20];
rz(1.5723751620236852) q[20];
h q[20];
cx q[17],q[20];
cx q[22],q[17];
rz(-4.595574239891973) q[17];
h q[17];
rz(1.5723751620236852) q[17];
h q[17];
cx q[22],q[17];
h q[22];
rz(1.5723751620236852) q[22];
h q[22];
cx q[23],q[7];
rz(-3.4379766976418833) q[7];
h q[7];
rz(0.8202907910159238) q[7];
h q[7];
rz(3*pi) q[7];
cx q[23],q[7];
cx q[23],q[18];
rz(-3.4592177043687466) q[18];
h q[18];
rz(0.8202907910159238) q[18];
h q[18];
rz(3*pi) q[18];
cx q[23],q[18];
h q[23];
rz(5.4628945161636615) q[23];
h q[23];
cx q[23],q[0];
rz(-4.445853947073388) q[0];
h q[0];
rz(1.5723751620236852) q[0];
h q[0];
cx q[23],q[0];
cx q[5],q[13];
rz(1.717724919120033) q[13];
cx q[5],q[13];
cx q[21],q[5];
rz(-4.349371981490111) q[5];
h q[5];
rz(1.5723751620236852) q[5];
h q[5];
cx q[21],q[5];
h q[21];
rz(1.5723751620236852) q[21];
h q[21];
cx q[7],q[10];
rz(1.6787581624161305) q[10];
cx q[7],q[10];
cx q[11],q[10];
rz(-4.351671014384769) q[10];
h q[10];
rz(1.5723751620236852) q[10];
h q[10];
cx q[11],q[10];
cx q[13],q[11];
rz(-4.3884744205445845) q[11];
h q[11];
rz(1.5723751620236852) q[11];
h q[11];
cx q[13],q[11];
h q[13];
rz(1.5723751620236852) q[13];
h q[13];
cx q[7],q[18];
rz(8.247028220475109) q[18];
cx q[7],q[18];
cx q[18],q[14];
rz(-4.628568805273299) q[14];
h q[14];
rz(1.5723751620236852) q[14];
h q[14];
cx q[18],q[14];
cx q[23],q[7];
rz(-4.468261861922728) q[7];
h q[7];
rz(1.5723751620236852) q[7];
h q[7];
cx q[23],q[7];
cx q[23],q[18];
rz(-4.474701160542322) q[18];
h q[18];
rz(1.5723751620236852) q[18];
h q[18];
cx q[23],q[18];
h q[23];
rz(1.5723751620236852) q[23];
h q[23];
