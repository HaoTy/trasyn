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
cx q[0],q[2];
rz(1.5532252469855101) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(1.61837082459956) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(1.4855910925926472) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(1.3849498604012627) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(1.281568010296186) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(1.2847155019358076) q[1];
cx q[7],q[1];
cx q[7],q[2];
rz(1.4898039415556499) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(1.3372059758326464) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(1.3070607445221578) q[3];
cx q[5],q[3];
cx q[4],q[6];
rz(1.428440218130531) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.5674828516416313) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(1.3380041769650055) q[5];
cx q[6],q[5];
rx(3.19743328403879) q[0];
rx(3.19743328403879) q[1];
rx(3.19743328403879) q[2];
rx(3.19743328403879) q[3];
rx(3.19743328403879) q[4];
rx(3.19743328403879) q[5];
rx(3.19743328403879) q[6];
rx(3.19743328403879) q[7];
cx q[0],q[2];
rz(6.222940534271378) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(6.4839439247002835) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(5.9519667513706915) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(5.548751310185795) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(5.13455568287711) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(5.147165993804966) q[1];
cx q[7],q[1];
cx q[7],q[2];
rz(5.968845377717716) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(5.357467170861996) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(5.236691396581284) q[3];
cx q[5],q[3];
cx q[4],q[6];
rz(5.722993848728532) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(6.2800631094473784) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(5.360665134705747) q[5];
cx q[6],q[5];
rx(2.296034505519087) q[0];
rx(2.296034505519087) q[1];
rx(2.296034505519087) q[2];
rx(2.296034505519087) q[3];
rx(2.296034505519087) q[4];
rx(2.296034505519087) q[5];
rx(2.296034505519087) q[6];
rx(2.296034505519087) q[7];
cx q[0],q[2];
rz(0.25806738038664295) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(0.26889127639996174) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(0.24683000893475154) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(0.23010853263830247) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(0.21293170443008147) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(0.21345465815092943) q[1];
cx q[7],q[1];
cx q[7],q[2];
rz(0.24752997109282018) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(0.22217591678363627) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(0.21716730590086467) q[3];
cx q[5],q[3];
cx q[4],q[6];
rz(0.23733442773178787) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(0.26043627227230765) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(0.22230853739075643) q[5];
cx q[6],q[5];
rx(2.890786233878528) q[0];
rx(2.890786233878528) q[1];
rx(2.890786233878528) q[2];
rx(2.890786233878528) q[3];
rx(2.890786233878528) q[4];
rx(2.890786233878528) q[5];
rx(2.890786233878528) q[6];
rx(2.890786233878528) q[7];
cx q[0],q[2];
rz(5.632068056587088) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(5.868288995835978) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(5.3868234204793435) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(5.021893562365888) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(4.647026094342359) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(4.658439047586773) q[1];
cx q[7],q[1];
cx q[7],q[2];
rz(5.40209940966237) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(4.84877198344605) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(4.739463951882863) q[3];
cx q[5],q[3];
cx q[4],q[6];
rz(5.179591652203106) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(5.68376680401798) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(4.851666298426454) q[5];
cx q[6],q[5];
rx(0.5103592639256677) q[0];
rx(0.5103592639256677) q[1];
rx(0.5103592639256677) q[2];
rx(0.5103592639256677) q[3];
rx(0.5103592639256677) q[4];
rx(0.5103592639256677) q[5];
rx(0.5103592639256677) q[6];
rx(0.5103592639256677) q[7];
cx q[0],q[2];
rz(3.312151165001729) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(3.451069844831235) q[5];
cx q[0],q[5];
cx q[6],q[0];
rz(3.1679257580937517) q[0];
cx q[6],q[0];
cx q[1],q[2];
rz(2.9533149184252325) q[2];
cx q[1],q[2];
cx q[1],q[3];
rz(2.732859890456737) q[3];
cx q[1],q[3];
cx q[7],q[1];
rz(2.7395717103432857) q[1];
cx q[7],q[1];
cx q[7],q[2];
rz(3.1769093827340718) q[2];
cx q[7],q[2];
cx q[3],q[4];
rz(2.8515042098995003) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(2.787221477438087) q[3];
cx q[5],q[3];
cx q[4],q[6];
rz(3.0460552594021144) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.342554573627462) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(2.853206321563974) q[5];
cx q[6],q[5];
rx(2.3408374136951493) q[0];
rx(2.3408374136951493) q[1];
rx(2.3408374136951493) q[2];
rx(2.3408374136951493) q[3];
rx(2.3408374136951493) q[4];
rx(2.3408374136951493) q[5];
rx(2.3408374136951493) q[6];
rx(2.3408374136951493) q[7];
