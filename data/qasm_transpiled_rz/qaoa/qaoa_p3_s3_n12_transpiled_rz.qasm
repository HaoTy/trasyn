OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[1],q[3];
rz(3.01380332630808) q[3];
cx q[1],q[3];
h q[4];
cx q[0],q[4];
rz(2.6146601548413884) q[4];
cx q[0],q[4];
h q[5];
cx q[0],q[5];
rz(3.0519398786144643) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(3.0150189677863763) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(-0.1906240852893193) q[5];
h q[5];
rz(1.2931715203735195) q[5];
h q[5];
rz(3*pi) q[5];
cx q[4],q[5];
h q[6];
cx q[6],q[0];
rz(-0.06401677706598896) q[0];
h q[0];
rz(1.2931715203735195) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(2.856870627991832) q[6];
cx q[2],q[6];
cx q[3],q[6];
rz(-0.04938155328789762) q[6];
h q[6];
rz(1.2931715203735195) q[6];
h q[6];
rz(3*pi) q[6];
cx q[3],q[6];
h q[7];
cx q[1],q[7];
rz(3.22261284773755) q[7];
cx q[1],q[7];
h q[8];
cx q[8],q[1];
rz(-0.2824621923478423) q[1];
h q[1];
rz(1.2931715203735195) q[1];
h q[1];
rz(3*pi) q[1];
cx q[8],q[1];
h q[9];
cx q[7],q[9];
rz(2.9136330615697275) q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(3.6628676304182948) q[9];
cx q[8],q[9];
h q[10];
cx q[10],q[2];
rz(-0.18522271740562157) q[2];
h q[2];
rz(1.2931715203735195) q[2];
h q[2];
rz(3*pi) q[2];
cx q[10],q[2];
cx q[10],q[7];
rz(-0.11185550785624976) q[7];
h q[7];
rz(1.2931715203735195) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[10],q[9];
rz(-0.1846529815121638) q[9];
h q[9];
rz(1.2931715203735195) q[9];
h q[9];
rz(3*pi) q[9];
cx q[10],q[9];
h q[10];
rz(4.990013786806067) q[10];
h q[10];
h q[11];
cx q[11],q[3];
rz(0.01619161991202933) q[3];
h q[3];
rz(1.2931715203735195) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[1],q[3];
rz(8.771314021087079) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(8.943702498288207) q[7];
cx q[1],q[7];
cx q[11],q[4];
rz(-0.012687856663378128) q[4];
h q[4];
rz(1.2931715203735195) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
cx q[0],q[4];
rz(8.441790333453831) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(8.80279870690405) q[5];
cx q[0],q[5];
cx q[11],q[8];
rz(-0.04567548277092737) q[8];
h q[8];
rz(1.2931715203735195) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
h q[11];
rz(4.990013786806067) q[11];
h q[11];
cx q[2],q[5];
rz(2.489132320361026) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(-3.846931566089804) q[5];
h q[5];
rz(0.687064560587316) q[5];
h q[5];
cx q[4],q[5];
cx q[6],q[0];
rz(-3.74240740015264) q[0];
h q[0];
rz(0.687064560587316) q[0];
h q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(2.358568583217091) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(-3.842472317402774) q[2];
h q[2];
rz(0.687064560587316) q[2];
h q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(-3.730324886274291) q[6];
h q[6];
rz(0.687064560587316) q[6];
h q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(-3.676189139035391) q[3];
h q[3];
rz(0.687064560587316) q[3];
h q[3];
cx q[11],q[3];
cx q[11],q[4];
rz(-3.700031389882567) q[4];
h q[4];
rz(0.6870645605873165) q[4];
h q[4];
cx q[11],q[4];
cx q[0],q[4];
rz(8.881097201642305) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(9.315575920987136) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(2.9957062006469797) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(-0.2095265762534506) q[5];
h q[5];
rz(1.5591513785135316) q[5];
h q[5];
rz(3*pi) q[5];
cx q[4],q[5];
cx q[6],q[0];
rz(-0.08373025379438559) q[0];
h q[0];
rz(1.5591513785135316) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(2.8385708833549694) q[6];
cx q[2],q[6];
cx q[7],q[9];
rz(8.688615687788559) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(-3.7819019878309765) q[7];
h q[7];
rz(0.687064560587316) q[7];
h q[7];
cx q[10],q[7];
cx q[8],q[1];
rz(-3.9227510561038192) q[1];
h q[1];
rz(0.687064560587316) q[1];
h q[1];
cx q[8],q[1];
cx q[1],q[3];
rz(9.277683653165216) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(9.485155640832232) q[7];
cx q[1],q[7];
cx q[3],q[6];
rz(-0.06918877624869779) q[6];
h q[6];
rz(1.5591513785135307) q[6];
h q[6];
rz(3*pi) q[6];
cx q[3],q[6];
cx q[8],q[9];
rz(3.0239817067460573) q[9];
cx q[8],q[9];
cx q[10],q[9];
rz(-3.8420019561739966) q[9];
h q[9];
rz(0.687064560587316) q[9];
h q[9];
cx q[10],q[9];
h q[10];
rz(0.687064560587316) q[10];
h q[10];
cx q[10],q[2];
rz(-0.20415980694484936) q[2];
h q[2];
rz(1.5591513785135307) q[2];
h q[2];
rz(3*pi) q[2];
cx q[10],q[2];
cx q[11],q[8];
rz(-3.7272652372293296) q[8];
h q[8];
rz(0.687064560587316) q[8];
h q[8];
cx q[11],q[8];
h q[11];
rz(0.687064560587316) q[11];
h q[11];
cx q[11],q[3];
rz(-0.004035633382951431) q[3];
h q[3];
rz(1.5591513785135307) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[11],q[4];
rz(-0.032730121866355155) q[4];
h q[4];
rz(1.5591513785135307) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(9.178155031155434) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(-0.13126255259271868) q[7];
h q[7];
rz(1.5591513785135307) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[8],q[1];
rz(-0.3007764123939154) q[1];
h q[1];
rz(1.5591513785135307) q[1];
h q[1];
rz(3*pi) q[1];
cx q[8],q[1];
cx q[8],q[9];
rz(3.639405055103009) q[9];
cx q[8],q[9];
cx q[10],q[9];
rz(-0.20359372050658742) q[9];
h q[9];
rz(1.5591513785135307) q[9];
h q[9];
rz(3*pi) q[9];
cx q[10],q[9];
h q[10];
rz(4.7240339286660555) q[10];
h q[10];
cx q[11],q[8];
rz(-0.06550644504403813) q[8];
h q[8];
rz(1.5591513785135307) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
h q[11];
rz(4.7240339286660555) q[11];
h q[11];
