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
cx q[0],q[3];
rz(1.352153368105036) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(1.243558134570985) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(1.191415340448219) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(1.2598093301423123) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(1.1687443392276742) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(1.3785097370466322) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(1.1597097116203967) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(1.2077007140310239) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(1.2410848603608853) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(1.119298100579319) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(0.9768621125272109) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(1.2104791740271779) q[6];
cx q[7],q[6];
rx(5.8055162522092445) q[0];
rx(5.8055162522092445) q[1];
rx(5.8055162522092445) q[2];
rx(5.8055162522092445) q[3];
rx(5.8055162522092445) q[4];
rx(5.8055162522092445) q[5];
rx(5.8055162522092445) q[6];
rx(5.8055162522092445) q[7];
cx q[0],q[3];
rz(0.4940740773509209) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(0.4543936009503503) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(0.43534073054053196) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(0.46033175460003983) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(0.4270567930265797) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(0.5037046406984717) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(0.4237555585626737) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(0.4412913727657668) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(0.45348987160853504) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(0.40898923847624274) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(0.35694341953409336) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(0.44230661637010743) q[6];
cx q[7],q[6];
rx(1.3964694629378986) q[0];
rx(1.3964694629378986) q[1];
rx(1.3964694629378986) q[2];
rx(1.3964694629378986) q[3];
rx(1.3964694629378986) q[4];
rx(1.3964694629378986) q[5];
rx(1.3964694629378986) q[6];
rx(1.3964694629378986) q[7];
cx q[0],q[3];
rz(3.6711773992232772) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(3.3763348344539437) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.234764024639579) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(3.4204578040068916) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(3.173210898151222) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(3.7427365198573366) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(3.148681342950898) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(3.278979789541225) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(3.369619746804586) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(3.038961397995156) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(2.652239157377022) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(3.2865234748827934) q[6];
cx q[7],q[6];
rx(0.945816753018065) q[0];
rx(0.945816753018065) q[1];
rx(0.945816753018065) q[2];
rx(0.945816753018065) q[3];
rx(0.945816753018065) q[4];
rx(0.945816753018065) q[5];
rx(0.945816753018065) q[6];
rx(0.945816753018065) q[7];
cx q[0],q[3];
rz(0.7335962858811262) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(0.674679108389183) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(0.6463895955230997) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(0.6834960199862463) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(0.6340896873285833) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(0.7478956507467882) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(0.6291880471645637) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(0.6552250500331713) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(0.6733372600328553) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(0.6072631616704321) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(0.5299860463108983) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(0.6567324736596083) q[6];
cx q[7],q[6];
rx(2.2620276097579537) q[0];
rx(2.2620276097579537) q[1];
rx(2.2620276097579537) q[2];
rx(2.2620276097579537) q[3];
rx(2.2620276097579537) q[4];
rx(2.2620276097579537) q[5];
rx(2.2620276097579537) q[6];
rx(2.2620276097579537) q[7];
cx q[0],q[3];
rz(4.327514706711905) q[3];
cx q[0],q[3];
cx q[0],q[5];
rz(3.9799598499311073) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.8130788483095572) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(4.03197117460438) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(3.740521183245804) q[3];
cx q[1],q[3];
cx q[4],q[1];
rz(4.411867249034916) q[1];
cx q[4],q[1];
cx q[2],q[4];
rz(3.711606206023248) q[4];
cx q[2],q[4];
cx q[5],q[2];
rz(3.8651995583904353) q[2];
cx q[5],q[2];
cx q[6],q[3];
rz(3.972044231207369) q[3];
cx q[6],q[3];
cx q[7],q[4];
rz(3.5822704034232284) q[4];
cx q[7],q[4];
cx q[7],q[5];
rz(3.1264095169289847) q[5];
cx q[7],q[5];
cx q[7],q[6];
rz(3.874091912452473) q[6];
cx q[7],q[6];
rx(1.7357228587258713) q[0];
rx(1.7357228587258713) q[1];
rx(1.7357228587258713) q[2];
rx(1.7357228587258713) q[3];
rx(1.7357228587258713) q[4];
rx(1.7357228587258713) q[5];
rx(1.7357228587258713) q[6];
rx(1.7357228587258713) q[7];
