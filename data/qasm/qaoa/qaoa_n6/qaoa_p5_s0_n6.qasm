OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0],q[1];
rz(6.2452826773250285) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(5.77642209983782) q[4];
cx q[0],q[4];
cx q[5],q[0];
rz(5.436881873194312) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(6.238099997111948) q[2];
cx q[1],q[2];
cx q[4],q[1];
rz(6.864881557906723) q[1];
cx q[4],q[1];
cx q[2],q[3];
rz(4.733937165865982) q[3];
cx q[2],q[3];
cx q[5],q[2];
rz(5.702060732331959) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(5.485342382939622) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(5.942733851295936) q[3];
cx q[5],q[3];
rx(0.4462164159191875) q[0];
rx(0.4462164159191875) q[1];
rx(0.4462164159191875) q[2];
rx(0.4462164159191875) q[3];
rx(0.4462164159191875) q[4];
rx(0.4462164159191875) q[5];
cx q[0],q[1];
rz(0.9019546032248914) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(0.8342409418928473) q[4];
cx q[0],q[4];
cx q[5],q[0];
rz(0.7852039508991413) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(0.9009172680366537) q[2];
cx q[1],q[2];
cx q[4],q[1];
rz(0.9914381528683185) q[1];
cx q[4],q[1];
cx q[2],q[3];
rz(0.6836834517727631) q[3];
cx q[2],q[3];
cx q[5],q[2];
rz(0.8235015436639209) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(0.7922027021323087) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(0.8582599747450627) q[3];
cx q[5],q[3];
rx(6.136080854923118) q[0];
rx(6.136080854923118) q[1];
rx(6.136080854923118) q[2];
rx(6.136080854923118) q[3];
rx(6.136080854923118) q[4];
rx(6.136080854923118) q[5];
cx q[0],q[1];
rz(5.064212789216575) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(4.684020273433256) q[4];
cx q[0],q[4];
cx q[5],q[0];
rz(4.40869183002034) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(5.058388453814112) q[2];
cx q[1],q[2];
cx q[4],q[1];
rz(5.566636896714311) q[1];
cx q[4],q[1];
cx q[2],q[3];
rz(3.8386837517808794) q[3];
cx q[2],q[3];
cx q[5],q[2];
rz(4.623721675626933) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(4.447987782806463) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(4.818880157644605) q[3];
cx q[5],q[3];
rx(0.6590142513698781) q[0];
rx(0.6590142513698781) q[1];
rx(0.6590142513698781) q[2];
rx(0.6590142513698781) q[3];
rx(0.6590142513698781) q[4];
rx(0.6590142513698781) q[5];
cx q[0],q[1];
rz(3.4498372978780107) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(3.190842983082191) q[4];
cx q[0],q[4];
cx q[5],q[0];
rz(3.0032840528423126) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(3.4458696507148514) q[2];
cx q[1],q[2];
cx q[4],q[1];
rz(3.792098079077707) q[1];
cx q[4],q[1];
cx q[2],q[3];
rz(2.6149837956751054) q[3];
cx q[2],q[3];
cx q[5],q[2];
rz(3.149766440610488) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(3.0300531972720024) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(3.2827120805913808) q[3];
cx q[5],q[3];
rx(1.0545414607865964) q[0];
rx(1.0545414607865964) q[1];
rx(1.0545414607865964) q[2];
rx(1.0545414607865964) q[3];
rx(1.0545414607865964) q[4];
rx(1.0545414607865964) q[5];
cx q[0],q[1];
rz(0.6745995590161901) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(0.6239544313006226) q[4];
cx q[0],q[4];
cx q[5],q[0];
rz(0.5872781591450641) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(0.6738237041582681) q[2];
cx q[1],q[2];
cx q[4],q[1];
rz(0.7415270550485018) q[1];
cx q[4],q[1];
cx q[2],q[3];
rz(0.5113478587764078) q[3];
cx q[2],q[3];
cx q[5],q[2];
rz(0.6159221054125683) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(0.5925127402189649) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(0.6419190039212025) q[3];
cx q[5],q[3];
rx(4.921597314153138) q[0];
rx(4.921597314153138) q[1];
rx(4.921597314153138) q[2];
rx(4.921597314153138) q[3];
rx(4.921597314153138) q[4];
rx(4.921597314153138) q[5];
