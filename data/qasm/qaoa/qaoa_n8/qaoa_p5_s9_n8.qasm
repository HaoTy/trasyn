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
rz(4.3326246015043175) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.385109946311133) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(4.8467625959436775) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(4.12485096942075) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(4.506109819394361) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(4.554229560669787) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(4.842584764509591) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(4.4648589708085415) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(4.332949454504388) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(4.6957694750254975) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(4.674408102087992) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(4.237522514275358) q[5];
cx q[6],q[5];
rx(5.907448377726257) q[0];
rx(5.907448377726257) q[1];
rx(5.907448377726257) q[2];
rx(5.907448377726257) q[3];
rx(5.907448377726257) q[4];
rx(5.907448377726257) q[5];
rx(5.907448377726257) q[6];
rx(5.907448377726257) q[7];
cx q[0],q[1];
rz(3.5517146477568113) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.5947400619314456) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(3.9731846835371) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(3.381390024609766) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.6939309822473203) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(3.733377646949095) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(3.9697598622185963) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.6601151646701915) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(3.551980949424931) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(3.849406482423296) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.831895271982403) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(3.4737537101472924) q[5];
cx q[6],q[5];
rx(4.694184427061165) q[0];
rx(4.694184427061165) q[1];
rx(4.694184427061165) q[2];
rx(4.694184427061165) q[3];
rx(4.694184427061165) q[4];
rx(4.694184427061165) q[5];
rx(4.694184427061165) q[6];
rx(4.694184427061165) q[7];
cx q[0],q[1];
rz(5.1016530320908515) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(5.163454374949186) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(5.7070490447834725) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(4.857000176697004) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.305931378191822) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(5.362592235422788) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(5.702129660261703) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(5.257358622386693) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(5.10203554556599) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(5.529255078306516) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(5.5041021229873754) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(4.989670597355717) q[5];
cx q[6],q[5];
rx(6.233725397929901) q[0];
rx(6.233725397929901) q[1];
rx(6.233725397929901) q[2];
rx(6.233725397929901) q[3];
rx(6.233725397929901) q[4];
rx(6.233725397929901) q[5];
rx(6.233725397929901) q[6];
rx(6.233725397929901) q[7];
cx q[0],q[1];
rz(3.5620459025269775) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.6051964693558967) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(3.984741913575405) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(3.3912258378115965) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.704675916980495) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(3.744237324442678) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(3.9813071301104195) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.6707617356942004) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(3.5623129788154286) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(3.8606036654821776) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.8430415182982345) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(3.483858191545033) q[5];
cx q[6],q[5];
rx(4.23345069104871) q[0];
rx(4.23345069104871) q[1];
rx(4.23345069104871) q[2];
rx(4.23345069104871) q[3];
rx(4.23345069104871) q[4];
rx(4.23345069104871) q[5];
rx(4.23345069104871) q[6];
rx(4.23345069104871) q[7];
cx q[0],q[1];
rz(3.5561063387472074) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.5991849537959344) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(3.9780975217317143) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(3.3855711094036898) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.6984985235682784) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(3.737993964024626) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(3.9746684656256726) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.6646408927724323) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(3.5563729696970268) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(3.854166269017613) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.8366334059855767) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(3.478049002529901) q[5];
cx q[6],q[5];
rx(4.152228263921584) q[0];
rx(4.152228263921584) q[1];
rx(4.152228263921584) q[2];
rx(4.152228263921584) q[3];
rx(4.152228263921584) q[4];
rx(4.152228263921584) q[5];
rx(4.152228263921584) q[6];
rx(4.152228263921584) q[7];
