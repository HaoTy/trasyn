OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[1];
rz(5.159312551534144) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(5.3667074094294245) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(5.114938255934356) q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(3.8857053586455823) q[4];
cx q[1],q[4];
cx q[7],q[1];
rz(4.958305617248491) q[1];
cx q[7],q[1];
cx q[2],q[3];
rz(4.860554216682636) q[3];
cx q[2],q[3];
cx q[2],q[5];
rz(4.934970101534604) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(4.753100902903555) q[2];
cx q[6],q[2];
cx q[7],q[3];
rz(4.877932351769289) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(4.936086988039525) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(4.802217855819053) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(4.707366939886124) q[5];
cx q[7],q[5];
rx(4.227040955584133) q[0];
rx(4.227040955584133) q[1];
rx(4.227040955584133) q[2];
rx(4.227040955584133) q[3];
rx(4.227040955584133) q[4];
rx(4.227040955584133) q[5];
rx(4.227040955584133) q[6];
rx(4.227040955584133) q[7];
cx q[0],q[1];
rz(2.0157234164335063) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(2.0967517835525147) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(1.998386551137055) q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(1.5181300226624235) q[4];
cx q[1],q[4];
cx q[7],q[1];
rz(1.9371907863092424) q[1];
cx q[7],q[1];
cx q[2],q[3];
rz(1.898999692991747) q[3];
cx q[2],q[3];
cx q[2],q[5];
rz(1.928073690768084) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(1.8570180997863766) q[2];
cx q[6],q[2];
cx q[7],q[3];
rz(1.9057892630126008) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(1.9285100539965145) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(1.876207902913576) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(1.8391500176999114) q[5];
cx q[7],q[5];
rx(0.7895014712586923) q[0];
rx(0.7895014712586923) q[1];
rx(0.7895014712586923) q[2];
rx(0.7895014712586923) q[3];
rx(0.7895014712586923) q[4];
rx(0.7895014712586923) q[5];
rx(0.7895014712586923) q[6];
rx(0.7895014712586923) q[7];
cx q[0],q[1];
rz(1.6884158509065081) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(1.7562870569963136) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(1.6738940976079575) q[0];
cx q[4],q[0];
cx q[1],q[4];
rz(1.2716202893229855) q[4];
cx q[1],q[4];
cx q[7],q[1];
rz(1.6226351309753027) q[1];
cx q[7],q[1];
cx q[2],q[3];
rz(1.5906454012360909) q[3];
cx q[2],q[3];
cx q[2],q[5];
rz(1.6149984440665617) q[5];
cx q[2],q[5];
cx q[6],q[2];
rz(1.5554806624448576) q[2];
cx q[6],q[2];
cx q[7],q[3];
rz(1.596332499749007) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(1.6153639518468592) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(1.57155447867956) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(1.540513949862104) q[5];
cx q[7],q[5];
rx(1.2435102573880572) q[0];
rx(1.2435102573880572) q[1];
rx(1.2435102573880572) q[2];
rx(1.2435102573880572) q[3];
rx(1.2435102573880572) q[4];
rx(1.2435102573880572) q[5];
rx(1.2435102573880572) q[6];
rx(1.2435102573880572) q[7];
