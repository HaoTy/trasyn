OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
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
cx q[0],q[1];
rz(6.05293930662209) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(5.18389794512834) q[7];
cx q[0],q[7];
cx q[8],q[0];
rz(4.543109679545274) q[0];
cx q[8],q[0];
cx q[1],q[2];
rz(5.11760352227932) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.266662495166913) q[1];
cx q[3],q[1];
cx q[2],q[4];
rz(4.882923684215409) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(5.379761263948972) q[2];
cx q[7],q[2];
cx q[3],q[5];
rz(5.66214044076181) q[5];
cx q[3],q[5];
cx q[6],q[3];
rz(5.537712262848976) q[3];
cx q[6],q[3];
cx q[4],q[5];
rz(4.227071358963117) q[5];
cx q[4],q[5];
cx q[6],q[4];
rz(5.334236234324877) q[4];
cx q[6],q[4];
cx q[9],q[5];
rz(5.53647753926706) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(5.208426572331408) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(5.05797163233811) q[7];
cx q[8],q[7];
cx q[9],q[8];
rz(5.191834118604998) q[8];
cx q[9],q[8];
rx(2.62052771518882) q[0];
rx(2.62052771518882) q[1];
rx(2.62052771518882) q[2];
rx(2.62052771518882) q[3];
rx(2.62052771518882) q[4];
rx(2.62052771518882) q[5];
rx(2.62052771518882) q[6];
rx(2.62052771518882) q[7];
rx(2.62052771518882) q[8];
rx(2.62052771518882) q[9];
cx q[0],q[1];
rz(7.246468057998394) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(6.206067659426039) q[7];
cx q[0],q[7];
cx q[8],q[0];
rz(5.438927686056768) q[0];
cx q[8],q[0];
cx q[1],q[2];
rz(6.126701190795949) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(6.3051518626218295) q[1];
cx q[3],q[1];
cx q[2],q[4];
rz(5.845746787614364) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(6.440551636824341) q[2];
cx q[7],q[2];
cx q[3],q[5];
rz(6.7786108145827315) q[5];
cx q[3],q[5];
cx q[6],q[3];
rz(6.629647679304993) q[3];
cx q[6],q[3];
cx q[4],q[5];
rz(5.060572397957885) q[5];
cx q[4],q[5];
cx q[6],q[4];
rz(6.386049905302017) q[4];
cx q[6],q[4];
cx q[9],q[5];
rz(6.628169490850828) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(6.235432882589457) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(6.0553109845164474) q[7];
cx q[8],q[7];
cx q[9],q[8];
rz(6.215568700934653) q[8];
cx q[9],q[8];
rx(4.506031742171101) q[0];
rx(4.506031742171101) q[1];
rx(4.506031742171101) q[2];
rx(4.506031742171101) q[3];
rx(4.506031742171101) q[4];
rx(4.506031742171101) q[5];
rx(4.506031742171101) q[6];
rx(4.506031742171101) q[7];
rx(4.506031742171101) q[8];
rx(4.506031742171101) q[9];
cx q[0],q[1];
rz(5.266495325534942) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(4.5103664373770505) q[7];
cx q[0],q[7];
cx q[8],q[0];
rz(3.952834264262605) q[0];
cx q[8],q[0];
cx q[1],q[2];
rz(4.452685490921594) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(4.582377586641555) q[1];
cx q[3],q[1];
cx q[2],q[4];
rz(4.248497045019166) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(4.680781701888974) q[2];
cx q[7],q[2];
cx q[3],q[5];
rz(4.926472025115284) q[5];
cx q[3],q[5];
cx q[6],q[3];
rz(4.818210503869588) q[3];
cx q[6],q[3];
cx q[4],q[5];
rz(3.6778580496134796) q[5];
cx q[4],q[5];
cx q[6],q[4];
rz(4.641171630886307) q[4];
cx q[6],q[4];
cx q[9],q[5];
rz(4.817136204980556) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(4.531708118494755) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(4.400801430348805) q[7];
cx q[8],q[7];
cx q[9],q[8];
rz(4.517271482744303) q[8];
cx q[9],q[8];
rx(4.350166454612852) q[0];
rx(4.350166454612852) q[1];
rx(4.350166454612852) q[2];
rx(4.350166454612852) q[3];
rx(4.350166454612852) q[4];
rx(4.350166454612852) q[5];
rx(4.350166454612852) q[6];
rx(4.350166454612852) q[7];
rx(4.350166454612852) q[8];
rx(4.350166454612852) q[9];
cx q[0],q[1];
rz(2.9995772118469897) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(2.568917571622732) q[7];
cx q[0],q[7];
cx q[8],q[0];
rz(2.2513703798048503) q[0];
cx q[8],q[0];
cx q[1],q[2];
rz(2.536064897908837) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(2.6099321342457) q[1];
cx q[3],q[1];
cx q[2],q[4];
rz(2.419767631625939) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(2.665979035154729) q[2];
cx q[7],q[2];
cx q[3],q[5];
rz(2.8059140487011622) q[5];
cx q[3],q[5];
cx q[6],q[3];
rz(2.744252778354266) q[3];
cx q[6],q[3];
cx q[4],q[5];
rz(2.094755337679529) q[5];
cx q[4],q[5];
cx q[6],q[4];
rz(2.6434187822740007) q[4];
cx q[6],q[4];
cx q[9],q[5];
rz(2.7436409022835457) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(2.5810729076453023) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(2.506513889860345) q[7];
cx q[8],q[7];
cx q[9],q[8];
rz(2.5728503989490865) q[8];
cx q[9],q[8];
rx(1.9289678890066941) q[0];
rx(1.9289678890066941) q[1];
rx(1.9289678890066941) q[2];
rx(1.9289678890066941) q[3];
rx(1.9289678890066941) q[4];
rx(1.9289678890066941) q[5];
rx(1.9289678890066941) q[6];
rx(1.9289678890066941) q[7];
rx(1.9289678890066941) q[8];
rx(1.9289678890066941) q[9];
cx q[0],q[1];
rz(4.962489803619221) q[1];
cx q[0],q[1];
cx q[0],q[7];
rz(4.250008036187985) q[7];
cx q[0],q[7];
cx q[8],q[0];
rz(3.724659098564259) q[0];
cx q[8],q[0];
cx q[1],q[2];
rz(4.195656690377338) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(4.31786238179793) q[1];
cx q[3],q[1];
cx q[2],q[4];
rz(4.003254909273558) q[4];
cx q[2],q[4];
cx q[7],q[2];
rz(4.410586174066726) q[2];
cx q[7],q[2];
cx q[3],q[5];
rz(4.642094159642433) q[5];
cx q[3],q[5];
cx q[6],q[3];
rz(4.540081974669783) q[3];
cx q[6],q[3];
cx q[4],q[5];
rz(3.4655557334064273) q[5];
cx q[4],q[5];
cx q[6],q[4];
rz(4.3732625724453) q[4];
cx q[6],q[4];
cx q[9],q[5];
rz(4.539069689088323) q[5];
cx q[9],q[5];
cx q[9],q[6];
rz(4.270117780599074) q[6];
cx q[9],q[6];
cx q[8],q[7];
rz(4.146767608426681) q[7];
cx q[8],q[7];
cx q[9],q[8];
rz(4.25651449164089) q[8];
cx q[9],q[8];
rx(4.578519416367647) q[0];
rx(4.578519416367647) q[1];
rx(4.578519416367647) q[2];
rx(4.578519416367647) q[3];
rx(4.578519416367647) q[4];
rx(4.578519416367647) q[5];
rx(4.578519416367647) q[6];
rx(4.578519416367647) q[7];
rx(4.578519416367647) q[8];
rx(4.578519416367647) q[9];
