OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
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
cx q[0],q[1];
rz(1.5733229346378077) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(1.2531806150158535) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(1.4467143337317285) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(1.332536469999806) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(1.0989406791962424) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(1.5083227224970321) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(1.7149658678923443) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(1.3499534542974987) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(1.4605554935056912) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(1.477317177497824) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(1.1160456655237152) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(1.4826818379249787) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(1.3422303789532652) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(1.4527735014865326) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(1.406217253586067) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(1.3757927987773684) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(1.4700902043279378) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(1.3055773836110554) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(0.9772996047827625) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(1.457508115589574) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(1.3042689667903755) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(1.3388302351643384) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(1.4676231874864405) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(1.4811826209814827) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(1.4752692418643567) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(1.5052876995073334) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(1.3022494471416324) q[16];
cx q[17],q[16];
rx(1.8467266085213652) q[0];
rx(1.8467266085213652) q[1];
rx(1.8467266085213652) q[2];
rx(1.8467266085213652) q[3];
rx(1.8467266085213652) q[4];
rx(1.8467266085213652) q[5];
rx(1.8467266085213652) q[6];
rx(1.8467266085213652) q[7];
rx(1.8467266085213652) q[8];
rx(1.8467266085213652) q[9];
rx(1.8467266085213652) q[10];
rx(1.8467266085213652) q[11];
rx(1.8467266085213652) q[12];
rx(1.8467266085213652) q[13];
rx(1.8467266085213652) q[14];
rx(1.8467266085213652) q[15];
rx(1.8467266085213652) q[16];
rx(1.8467266085213652) q[17];
cx q[0],q[1];
rz(4.147651799493818) q[1];
cx q[0],q[1];
cx q[0],q[8];
rz(3.303680839151971) q[8];
cx q[0],q[8];
cx q[13],q[0];
rz(3.8138815481242965) q[0];
cx q[13],q[0];
cx q[1],q[11];
rz(3.5128816633936464) q[11];
cx q[1],q[11];
cx q[14],q[1];
rz(2.8970678461854042) q[1];
cx q[14],q[1];
cx q[2],q[4];
rz(3.9762958490288667) q[4];
cx q[2],q[4];
cx q[2],q[12];
rz(4.521056110881426) q[12];
cx q[2],q[12];
cx q[17],q[2];
rz(3.5587969581330006) q[2];
cx q[17],q[2];
cx q[3],q[7];
rz(3.8503701227072216) q[7];
cx q[3],q[7];
cx q[3],q[11];
rz(3.894557890673955) q[11];
cx q[3],q[11];
cx q[13],q[3];
rz(2.9421606403979235) q[3];
cx q[13],q[3];
cx q[4],q[6];
rz(3.908700405846458) q[6];
cx q[4],q[6];
cx q[9],q[4];
rz(3.5384371027950303) q[4];
cx q[9],q[4];
cx q[5],q[8];
rz(3.8298549490633946) q[8];
cx q[5],q[8];
cx q[5],q[9];
rz(3.7071216556429314) q[9];
cx q[5],q[9];
cx q[14],q[5];
rz(3.626915588625312) q[5];
cx q[14],q[5];
cx q[6],q[15];
rz(3.8755058781385423) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(3.4418111280882218) q[6];
cx q[16],q[6];
cx q[7],q[9];
rz(2.576393170900398) q[9];
cx q[7],q[9];
cx q[11],q[7];
rz(3.842336512938206) q[7];
cx q[11],q[7];
cx q[15],q[8];
rz(3.438361831531676) q[8];
cx q[15],q[8];
cx q[10],q[12];
rz(3.52947351865461) q[12];
cx q[10],q[12];
cx q[10],q[16];
rz(3.869002237584689) q[16];
cx q[10],q[16];
cx q[17],q[10];
rz(3.904748114987014) q[10];
cx q[17],q[10];
cx q[15],q[12];
rz(3.889159047417816) q[12];
cx q[15],q[12];
cx q[14],q[13];
rz(3.968294809771388) q[13];
cx q[14],q[13];
cx q[17],q[16];
rz(3.43303789953983) q[16];
cx q[17],q[16];
rx(3.9052759942012862) q[0];
rx(3.9052759942012862) q[1];
rx(3.9052759942012862) q[2];
rx(3.9052759942012862) q[3];
rx(3.9052759942012862) q[4];
rx(3.9052759942012862) q[5];
rx(3.9052759942012862) q[6];
rx(3.9052759942012862) q[7];
rx(3.9052759942012862) q[8];
rx(3.9052759942012862) q[9];
rx(3.9052759942012862) q[10];
rx(3.9052759942012862) q[11];
rx(3.9052759942012862) q[12];
rx(3.9052759942012862) q[13];
rx(3.9052759942012862) q[14];
rx(3.9052759942012862) q[15];
rx(3.9052759942012862) q[16];
rx(3.9052759942012862) q[17];
