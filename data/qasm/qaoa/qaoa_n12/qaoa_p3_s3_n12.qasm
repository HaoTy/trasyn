OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
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
cx q[0],q[4];
rz(2.6146601548413884) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(3.0519398786144643) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.0775758765238037) q[0];
cx q[6],q[0];
cx q[1],q[3];
rz(3.01380332630808) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(3.22261284773755) q[7];
cx q[1],q[7];
cx q[8],q[1];
rz(2.859130461241951) q[1];
cx q[8],q[1];
cx q[2],q[5];
rz(3.0150189677863763) q[5];
cx q[2],q[5];
cx q[2],q[6];
rz(2.856870627991832) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(2.956369936184171) q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(3.092211100301896) q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(3.157784273501823) q[3];
cx q[11],q[3];
cx q[4],q[5];
rz(2.950968568300474) q[5];
cx q[4],q[5];
cx q[11],q[4];
rz(3.1289047969264154) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(2.9136330615697275) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(3.0297371457335434) q[7];
cx q[10],q[7];
cx q[8],q[9];
rz(3.6628676304182948) q[9];
cx q[8],q[9];
cx q[11],q[8];
rz(3.0959171708188657) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(2.9569396720776293) q[9];
cx q[10],q[9];
rx(4.990013786806067) q[0];
rx(4.990013786806067) q[1];
rx(4.990013786806067) q[2];
rx(4.990013786806067) q[3];
rx(4.990013786806067) q[4];
rx(4.990013786806067) q[5];
rx(4.990013786806067) q[6];
rx(4.990013786806067) q[7];
rx(4.990013786806067) q[8];
rx(4.990013786806067) q[9];
rx(4.990013786806067) q[10];
rx(4.990013786806067) q[11];
cx q[0],q[4];
rz(2.158605026274247) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(2.5196133997244616) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(2.5407779070269463) q[0];
cx q[6],q[0];
cx q[1],q[3];
rz(2.4881287139074915) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(2.6605171911086205) q[7];
cx q[1],q[7];
cx q[8],q[1];
rz(2.360434251075767) q[1];
cx q[8],q[1];
cx q[2],q[5];
rz(2.489132320361026) q[5];
cx q[2],q[5];
cx q[2],q[6];
rz(2.358568583217091) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(2.4407129897768116) q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(2.5528604209052954) q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(2.606996168144195) q[3];
cx q[11],q[3];
cx q[4],q[5];
rz(2.436253741089782) q[5];
cx q[4],q[5];
cx q[11],q[4];
rz(2.583153917297019) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(2.4054303806089727) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(2.5012833193486097) q[7];
cx q[10],q[7];
cx q[8],q[9];
rz(3.0239817067460573) q[9];
cx q[8],q[9];
cx q[11],q[8];
rz(2.5559200699502567) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(2.4411833510055896) q[9];
cx q[10],q[9];
rx(0.6870645605873161) q[0];
rx(0.6870645605873161) q[1];
rx(0.6870645605873161) q[2];
rx(0.6870645605873161) q[3];
rx(0.6870645605873161) q[4];
rx(0.6870645605873161) q[5];
rx(0.6870645605873161) q[6];
rx(0.6870645605873161) q[7];
rx(0.6870645605873161) q[8];
rx(0.6870645605873161) q[9];
rx(0.6870645605873161) q[10];
rx(0.6870645605873161) q[11];
cx q[0],q[4];
rz(2.5979118944627198) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(3.032390613807552) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.0578623997954075) q[0];
cx q[6],q[0];
cx q[1],q[3];
rz(2.9944983459856305) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(3.201970333652646) q[7];
cx q[1],q[7];
cx q[8],q[1];
rz(2.840816241195878) q[1];
cx q[8],q[1];
cx q[2],q[5];
rz(2.9957062006469797) q[5];
cx q[2],q[5];
cx q[2],q[6];
rz(2.8385708833549694) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(2.937432846644944) q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(3.072403877341095) q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(3.1375570202068412) q[3];
cx q[11],q[3];
cx q[4],q[5];
rz(2.9320660773363425) q[5];
cx q[4],q[5];
cx q[11],q[4];
rz(3.108862531723438) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(2.894969723975849) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(3.0103301009970744) q[7];
cx q[10],q[7];
cx q[8],q[9];
rz(3.639405055103009) q[9];
cx q[8],q[9];
cx q[11],q[8];
rz(3.076086208545755) q[8];
cx q[11],q[8];
cx q[10],q[9];
rz(2.9379989330832053) q[9];
cx q[10],q[9];
rx(4.7240339286660555) q[0];
rx(4.7240339286660555) q[1];
rx(4.7240339286660555) q[2];
rx(4.7240339286660555) q[3];
rx(4.7240339286660555) q[4];
rx(4.7240339286660555) q[5];
rx(4.7240339286660555) q[6];
rx(4.7240339286660555) q[7];
rx(4.7240339286660555) q[8];
rx(4.7240339286660555) q[9];
rx(4.7240339286660555) q[10];
rx(4.7240339286660555) q[11];
