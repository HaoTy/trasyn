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
cx q[0],q[12];
rz(2.5341302106493018) q[12];
cx q[0],q[12];
cx q[0],q[21];
rz(2.3628865331331923) q[21];
cx q[0],q[21];
cx q[22],q[0];
rz(2.2672028192663025) q[0];
cx q[22],q[0];
cx q[1],q[3];
rz(2.001661693571214) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(2.461050190678968) q[5];
cx q[1],q[5];
cx q[11],q[1];
rz(2.192738817536033) q[1];
cx q[11],q[1];
cx q[2],q[6];
rz(2.6735079331106943) q[6];
cx q[2],q[6];
cx q[2],q[10];
rz(2.6558610800773805) q[10];
cx q[2],q[10];
cx q[15],q[2];
rz(2.475168472553502) q[2];
cx q[15],q[2];
cx q[3],q[4];
rz(2.473346006699457) q[4];
cx q[3],q[4];
cx q[12],q[3];
rz(2.2923644131993424) q[3];
cx q[12],q[3];
cx q[4],q[11];
rz(2.546790934827944) q[11];
cx q[4],q[11];
cx q[19],q[4];
rz(2.2249268947430285) q[4];
cx q[19],q[4];
cx q[5],q[7];
rz(2.6707061802965204) q[7];
cx q[5],q[7];
cx q[14],q[5];
rz(2.604007246239521) q[5];
cx q[14],q[5];
cx q[6],q[11];
rz(2.4557296804916335) q[11];
cx q[6],q[11];
cx q[23],q[6];
rz(2.221694324856265) q[6];
cx q[23],q[6];
cx q[7],q[8];
rz(2.0020852221933185) q[8];
cx q[7],q[8];
cx q[19],q[7];
rz(2.578832223325123) q[7];
cx q[19],q[7];
cx q[8],q[9];
rz(2.6780513898195397) q[9];
cx q[8],q[9];
cx q[15],q[8];
rz(2.475641057096831) q[8];
cx q[15],q[8];
cx q[9],q[10];
rz(2.395409478415999) q[10];
cx q[9],q[10];
cx q[16],q[9];
rz(2.3651641755148383) q[9];
cx q[16],q[9];
cx q[13],q[10];
rz(2.6032313156953215) q[10];
cx q[13],q[10];
cx q[18],q[12];
rz(2.915968720673421) q[12];
cx q[18],q[12];
cx q[13],q[17];
rz(2.507991677096947) q[17];
cx q[13],q[17];
cx q[22],q[13];
rz(2.624888389651845) q[13];
cx q[22],q[13];
cx q[14],q[15];
rz(2.5130917735488105) q[15];
cx q[14],q[15];
cx q[20],q[14];
rz(2.474099021438182) q[14];
cx q[20],q[14];
cx q[16],q[17];
rz(2.539617589098406) q[17];
cx q[16],q[17];
cx q[18],q[16];
rz(2.6729171410009323) q[16];
cx q[18],q[16];
cx q[21],q[17];
rz(2.251623165445588) q[17];
cx q[21],q[17];
cx q[20],q[18];
rz(2.4975688751963845) q[18];
cx q[20],q[18];
cx q[23],q[19];
rz(2.2141493090627318) q[19];
cx q[23],q[19];
cx q[22],q[20];
rz(2.546853808876441) q[20];
cx q[22],q[20];
cx q[23],q[21];
rz(2.4457179387044437) q[21];
cx q[23],q[21];
rx(5.395950399005146) q[0];
rx(5.395950399005146) q[1];
rx(5.395950399005146) q[2];
rx(5.395950399005146) q[3];
rx(5.395950399005146) q[4];
rx(5.395950399005146) q[5];
rx(5.395950399005146) q[6];
rx(5.395950399005146) q[7];
rx(5.395950399005146) q[8];
rx(5.395950399005146) q[9];
rx(5.395950399005146) q[10];
rx(5.395950399005146) q[11];
rx(5.395950399005146) q[12];
rx(5.395950399005146) q[13];
rx(5.395950399005146) q[14];
rx(5.395950399005146) q[15];
rx(5.395950399005146) q[16];
rx(5.395950399005146) q[17];
rx(5.395950399005146) q[18];
rx(5.395950399005146) q[19];
rx(5.395950399005146) q[20];
rx(5.395950399005146) q[21];
rx(5.395950399005146) q[22];
rx(5.395950399005146) q[23];
cx q[0],q[12];
rz(5.7130058833543975) q[12];
cx q[0],q[12];
cx q[0],q[21];
rz(5.326949897349594) q[21];
cx q[0],q[21];
cx q[22],q[0];
rz(5.111238164003943) q[0];
cx q[22],q[0];
cx q[1],q[3];
rz(4.512595676339551) q[3];
cx q[1],q[3];
cx q[1],q[5];
rz(5.548252477119917) q[5];
cx q[1],q[5];
cx q[11],q[1];
rz(4.943364675027171) q[1];
cx q[11],q[1];
cx q[2],q[6];
rz(6.02722247139091) q[6];
cx q[2],q[6];
cx q[2],q[10];
rz(5.987438969036396) q[10];
cx q[2],q[10];
cx q[15],q[2];
rz(5.580081081298628) q[2];
cx q[15],q[2];
cx q[3],q[4];
rz(5.575972469159197) q[4];
cx q[3],q[4];
cx q[12],q[3];
rz(5.16796308428229) q[3];
cx q[12],q[3];
cx q[4],q[11];
rz(5.741548533379306) q[11];
cx q[4],q[11];
cx q[19],q[4];
rz(5.015930273150211) q[4];
cx q[19],q[4];
cx q[5],q[7];
rz(6.020906130484742) q[7];
cx q[5],q[7];
cx q[14],q[5];
rz(5.870538402307321) q[5];
cx q[14],q[5];
cx q[6],q[11];
rz(5.536257787235809) q[11];
cx q[6],q[11];
cx q[23],q[6];
rz(5.008642687570028) q[6];
cx q[23],q[6];
cx q[7],q[8];
rz(4.5135504897503544) q[8];
cx q[7],q[8];
cx q[19],q[7];
rz(5.813783207400944) q[7];
cx q[19],q[7];
cx q[8],q[9];
rz(6.0374653526759055) q[9];
cx q[8],q[9];
cx q[15],q[8];
rz(5.581146487592698) q[8];
cx q[15],q[8];
cx q[9],q[10];
rz(5.400270430352941) q[10];
cx q[9],q[10];
cx q[16],q[9];
rz(5.332084670721476) q[9];
cx q[16],q[9];
cx q[13],q[10];
rz(5.868789125279069) q[10];
cx q[13],q[10];
cx q[18],q[12];
rz(6.573832073378838) q[12];
cx q[18],q[12];
cx q[13],q[17];
rz(5.654078526212557) q[17];
cx q[13],q[17];
cx q[22],q[13];
rz(5.917613368962333) q[13];
cx q[22],q[13];
cx q[14],q[15];
rz(5.665576309914724) q[15];
cx q[14],q[15];
cx q[20],q[14];
rz(5.577670084228269) q[14];
cx q[20],q[14];
cx q[16],q[17];
rz(5.725376765178932) q[17];
cx q[16],q[17];
cx q[18],q[16];
rz(6.025890575032655) q[16];
cx q[18],q[16];
cx q[21],q[17];
rz(5.076115006731151) q[17];
cx q[21],q[17];
cx q[20],q[18];
rz(5.630581103574714) q[18];
cx q[20],q[18];
cx q[23],q[19];
rz(4.991633017176093) q[19];
cx q[23],q[19];
cx q[22],q[20];
rz(5.741690278190786) q[20];
cx q[22],q[20];
cx q[23],q[21];
rz(5.513687068694008) q[21];
cx q[23],q[21];
rx(2.360635094217161) q[0];
rx(2.360635094217161) q[1];
rx(2.360635094217161) q[2];
rx(2.360635094217161) q[3];
rx(2.360635094217161) q[4];
rx(2.360635094217161) q[5];
rx(2.360635094217161) q[6];
rx(2.360635094217161) q[7];
rx(2.360635094217161) q[8];
rx(2.360635094217161) q[9];
rx(2.360635094217161) q[10];
rx(2.360635094217161) q[11];
rx(2.360635094217161) q[12];
rx(2.360635094217161) q[13];
rx(2.360635094217161) q[14];
rx(2.360635094217161) q[15];
rx(2.360635094217161) q[16];
rx(2.360635094217161) q[17];
rx(2.360635094217161) q[18];
rx(2.360635094217161) q[19];
rx(2.360635094217161) q[20];
rx(2.360635094217161) q[21];
rx(2.360635094217161) q[22];
rx(2.360635094217161) q[23];
