OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.16243571460638487) q[2];
cx q[1],q[2];
h q[3];
cx q[0],q[3];
rz(0.1651869919652294) q[3];
cx q[0],q[3];
h q[4];
cx q[0],q[4];
rz(0.170217721661328) q[4];
cx q[0],q[4];
cx q[2],q[4];
rz(0.14583529538630696) q[4];
cx q[2],q[4];
h q[5];
cx q[5],q[2];
rz(-2.993509394547692) q[2];
h q[2];
rz(2.82568050980999) q[2];
h q[2];
rz(3*pi) q[2];
cx q[5],q[2];
cx q[3],q[5];
rz(0.1410942768092088) q[5];
cx q[3],q[5];
h q[6];
cx q[6],q[0];
rz(-2.9644610278438353) q[0];
h q[0];
rz(2.82568050980999) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[1],q[6];
rz(0.1835935284839565) q[6];
cx q[1],q[6];
h q[7];
cx q[7],q[5];
rz(-3.0065286610453685) q[5];
h q[5];
rz(2.82568050980999) q[5];
h q[5];
rz(3*pi) q[5];
cx q[7],q[5];
h q[8];
cx q[8],q[3];
rz(-3.008101592434233) q[3];
h q[3];
rz(2.82568050980999) q[3];
h q[3];
rz(3*pi) q[3];
cx q[8],q[3];
cx q[0],q[3];
rz(9.122055851714965) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[4];
rz(-2.984038379835937) q[4];
h q[4];
rz(2.82568050980999) q[4];
h q[4];
rz(3*pi) q[4];
cx q[9],q[4];
cx q[0],q[4];
rz(9.208512964238572) q[4];
cx q[0],q[4];
cx q[9],q[6];
rz(-2.9745668294448198) q[6];
h q[6];
rz(2.82568050980999) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
cx q[6],q[0];
rz(-0.09744402630737525) q[0];
h q[0];
rz(2.7170366085872253) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[7],q[9];
rz(-2.980241958876822) q[9];
h q[9];
rz(2.82568050980999) q[9];
h q[9];
rz(3*pi) q[9];
cx q[7],q[9];
h q[10];
cx q[10],q[7];
rz(-2.9599233710853747) q[7];
h q[7];
rz(2.82568050980999) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[8],q[10];
rz(0.1630000096724429) q[10];
cx q[8],q[10];
h q[11];
cx q[11],q[1];
rz(-2.973372636713325) q[1];
h q[1];
rz(2.82568050980999) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[1],q[2];
rz(9.074772950313655) q[2];
cx q[1],q[2];
cx q[1],q[6];
cx q[11],q[8];
rz(-2.958820171129619) q[8];
h q[8];
rz(2.82568050980999) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
cx q[11],q[10];
rz(-2.9991352977596524) q[10];
h q[10];
rz(2.82568050980999) q[10];
h q[10];
rz(3*pi) q[10];
cx q[11],q[10];
h q[11];
rz(3.4575047973695963) q[11];
h q[11];
cx q[2],q[4];
rz(2.506296164730383) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(-0.5966634359071321) q[2];
h q[2];
rz(2.7170366085872253) q[2];
h q[2];
rz(3*pi) q[2];
cx q[5],q[2];
cx q[3],q[5];
rz(2.424817969446991) q[5];
cx q[3],q[5];
rz(3.155201592932507) q[6];
cx q[1],q[6];
cx q[11],q[1];
rz(-0.25059715067288657) q[1];
h q[1];
rz(2.717036608587226) q[1];
h q[1];
rz(3*pi) q[1];
cx q[11],q[1];
cx q[7],q[5];
rz(-0.8204099411978723) q[5];
h q[5];
rz(2.7170366085872253) q[5];
h q[5];
rz(3*pi) q[5];
cx q[7],q[5];
cx q[8],q[3];
rz(-0.8474420248704373) q[3];
h q[3];
rz(2.7170366085872253) q[3];
h q[3];
rz(3*pi) q[3];
cx q[8],q[3];
cx q[9],q[4];
rz(-0.4338964748275238) q[4];
h q[4];
rz(2.7170366085872253) q[4];
h q[4];
rz(3*pi) q[4];
cx q[9],q[4];
cx q[9],q[6];
rz(-0.2711203076697295) q[6];
h q[6];
rz(2.7170366085872253) q[6];
h q[6];
rz(3*pi) q[6];
cx q[9],q[6];
cx q[7],q[9];
rz(-0.3686519449328305) q[9];
h q[9];
rz(2.7170366085872253) q[9];
h q[9];
rz(3*pi) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(-0.01946076649042805) q[7];
h q[7];
rz(2.7170366085872253) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[8],q[10];
rz(2.801285505068597) q[10];
cx q[8],q[10];
cx q[11],q[8];
rz(-0.0005013931047344045) q[8];
h q[8];
rz(2.7170366085872253) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
cx q[11],q[10];
rz(-0.693349081216315) q[10];
h q[10];
rz(2.7170366085872253) q[10];
h q[10];
rz(3*pi) q[10];
cx q[11],q[10];
h q[11];
rz(3.566148698592361) q[11];
h q[11];
