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
rz(4.950908716415627) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(4.674678257690674) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(4.705603127766985) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(4.4615492412328415) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(4.2565073506119715) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(4.197845794377924) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(4.666457851285864) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(4.441116471115146) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(4.631713500951733) q[3];
cx q[5],q[3];
rx(5.649135000081464) q[0];
rx(5.649135000081464) q[1];
rx(5.649135000081464) q[2];
rx(5.649135000081464) q[3];
rx(5.649135000081464) q[4];
rx(5.649135000081464) q[5];
cx q[0],q[1];
rz(3.021358688276851) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(2.852785332507213) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(2.871657650750345) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(2.722720481288349) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(2.597590907472133) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(2.5617919031382583) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(2.847768718843949) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(2.710251108278911) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(2.8265655113595334) q[3];
cx q[5],q[3];
rx(5.196810081244653) q[0];
rx(5.196810081244653) q[1];
rx(5.196810081244653) q[2];
rx(5.196810081244653) q[3];
rx(5.196810081244653) q[4];
rx(5.196810081244653) q[5];
cx q[0],q[1];
rz(4.14669496434106) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(3.915334719625892) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(3.941236227887022) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(3.736825870054434) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(3.5650904929715583) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(3.5159577794093813) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(3.908449616345631) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(3.7197121501682933) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(3.8793490550502594) q[3];
cx q[5],q[3];
rx(6.068861868376928) q[0];
rx(6.068861868376928) q[1];
rx(6.068861868376928) q[2];
rx(6.068861868376928) q[3];
rx(6.068861868376928) q[4];
rx(6.068861868376928) q[5];
cx q[0],q[1];
rz(0.09519732316307977) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(0.08988589414005323) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(0.09048052535203045) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(0.08578779558535905) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(0.08184519832868098) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(0.08071723911029635) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(0.08972783008961822) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(0.08539490912119102) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(0.08905975694662475) q[3];
cx q[5],q[3];
rx(5.859269397715153) q[0];
rx(5.859269397715153) q[1];
rx(5.859269397715153) q[2];
rx(5.859269397715153) q[3];
rx(5.859269397715153) q[4];
rx(5.859269397715153) q[5];
cx q[0],q[1];
rz(4.990970803164191) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(4.7125051247585885) q[3];
cx q[0],q[3];
cx q[4],q[0];
rz(4.743680234719868) q[0];
cx q[4],q[0];
cx q[1],q[2];
rz(4.497651496995025) q[2];
cx q[1],q[2];
cx q[5],q[1];
rz(4.2909504350018555) q[1];
cx q[5],q[1];
cx q[2],q[4];
rz(4.231814197352887) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(4.704218199931936) q[2];
cx q[5],q[2];
cx q[3],q[4];
rz(4.477053387652581) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(4.669192702992841) q[3];
cx q[5],q[3];
rx(3.3278157447570273) q[0];
rx(3.3278157447570273) q[1];
rx(3.3278157447570273) q[2];
rx(3.3278157447570273) q[3];
rx(3.3278157447570273) q[4];
rx(3.3278157447570273) q[5];
