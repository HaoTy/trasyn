OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[1],q[3];
rz(3.588129488500612) q[3];
cx q[1],q[3];
h q[4];
cx q[0],q[4];
rz(3.360774646403656) q[4];
cx q[0],q[4];
h q[5];
cx q[0],q[5];
rz(3.767967920329998) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(3.387470811015776) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(0.5688721230827403) q[5];
h q[5];
rz(0.6940567798217696) q[5];
h q[5];
rz(3*pi) q[5];
cx q[4],q[5];
h q[6];
cx q[6],q[0];
rz(-0.0590934981254696) q[0];
h q[0];
rz(0.69405677982177) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(3.2781730602619206) q[6];
cx q[2],q[6];
cx q[3],q[6];
rz(0.3754601652125018) q[6];
h q[6];
rz(0.6940567798217696) q[6];
h q[6];
rz(3*pi) q[6];
cx q[3],q[6];
h q[7];
cx q[1],q[7];
rz(3.922249679191425) q[7];
cx q[1],q[7];
h q[8];
cx q[8],q[1];
rz(0.945909166480174) q[1];
h q[1];
rz(0.6940567798217696) q[1];
h q[1];
rz(3*pi) q[1];
cx q[8],q[1];
h q[9];
cx q[7],q[9];
rz(3.5869700188940374) q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(3.4370605101144225) q[9];
cx q[8],q[9];
h q[10];
cx q[10],q[2];
rz(0.6784473416465788) q[2];
h q[2];
rz(0.6940567798217696) q[2];
h q[2];
rz(3*pi) q[2];
cx q[10],q[2];
cx q[10],q[7];
rz(0.4820386506268237) q[7];
h q[7];
rz(0.6940567798217696) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[10],q[9];
rz(0.0322187478716911) q[9];
h q[9];
rz(0.6940567798217696) q[9];
h q[9];
rz(3*pi) q[9];
cx q[10],q[9];
h q[10];
rz(5.589128527357817) q[10];
h q[10];
h q[11];
cx q[11],q[3];
rz(0.5584838963726364) q[3];
h q[3];
rz(0.6940567798217696) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[1],q[3];
rz(7.0521145546113155) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(7.123715868778768) q[7];
cx q[1],q[7];
cx q[11],q[4];
rz(0.281000308732005) q[4];
h q[4];
rz(0.6940567798217696) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
cx q[0],q[4];
rz(7.003392849278818) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(7.090653584623762) q[5];
cx q[0],q[5];
cx q[11],q[8];
rz(0.11775533688633999) q[8];
h q[8];
rz(0.6940567798217696) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
h q[11];
rz(5.589128527357817) q[11];
h q[11];
cx q[2],q[5];
rz(0.7259284788241465) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(0.7951454643686748) q[5];
h q[5];
rz(0.06792444743325188) q[5];
h q[5];
cx q[4],q[5];
cx q[6],q[0];
rz(0.6605736396683337) q[0];
h q[0];
rz(0.06792444743325188) q[0];
h q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(0.7025061810775114) q[6];
cx q[2],q[6];
cx q[10],q[2];
rz(0.818627222933265) q[2];
h q[2];
rz(0.06792444743325188) q[2];
h q[2];
cx q[10],q[2];
cx q[3],q[6];
rz(0.753697653835057) q[6];
h q[6];
rz(0.06792444743325188) q[6];
h q[6];
cx q[3],q[6];
cx q[11],q[3];
rz(0.7929192873669937) q[3];
h q[3];
rz(0.06792444743325188) q[3];
h q[3];
cx q[11],q[3];
cx q[11],q[4];
rz(0.7334550882897393) q[4];
h q[4];
rz(0.06792444743325188) q[4];
h q[4];
cx q[11],q[4];
cx q[0],q[4];
rz(6.541881590502464) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(6.573225372112916) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(0.26075122579619275) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(0.2856137492446287) q[5];
h q[5];
rz(1.7531878769300802) q[5];
h q[5];
cx q[4],q[5];
cx q[6],q[0];
rz(0.23727597318012972) q[0];
h q[0];
rz(1.753187876930081) q[0];
h q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(0.2523380101330035) q[6];
cx q[2],q[6];
cx q[7],q[9];
rz(7.051866082504929) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(0.7765372182500787) q[7];
h q[7];
rz(0.06792444743325188) q[7];
h q[7];
cx q[10],q[7];
cx q[8],q[1];
rz(0.8759437775183541) q[1];
h q[1];
rz(0.06792444743325188) q[1];
h q[1];
cx q[8],q[1];
cx q[1],q[3];
rz(6.559382273777637) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(6.585101240566245) q[7];
cx q[1],q[7];
cx q[3],q[6];
rz(0.2707258260972756) q[6];
h q[6];
rz(1.7531878769300802) q[6];
h q[6];
cx q[3],q[6];
cx q[8],q[9];
rz(0.7365554559526174) q[9];
cx q[8],q[9];
cx q[10],q[9];
rz(0.6801416783416645) q[9];
h q[9];
rz(0.06792444743325188) q[9];
h q[9];
cx q[10],q[9];
h q[10];
rz(0.06792444743325188) q[10];
h q[10];
cx q[10],q[2];
rz(0.29404832304656203) q[2];
h q[2];
rz(1.7531878769300802) q[2];
h q[2];
cx q[10],q[2];
cx q[11],q[8];
rz(0.6984720048334196) q[8];
h q[8];
rz(0.06792444743325188) q[8];
h q[8];
cx q[11],q[8];
h q[11];
rz(0.06792444743325188) q[11];
h q[11];
cx q[11],q[3];
rz(0.28481411346925967) q[3];
h q[3];
rz(1.7531878769300802) q[3];
h q[3];
cx q[11],q[3];
cx q[11],q[4];
rz(0.2634547602372468) q[4];
h q[4];
rz(1.7531878769300802) q[4];
h q[4];
cx q[11],q[4];
cx q[0],q[4];
rz(10.03652615956701) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(10.491282967340656) q[5];
cx q[0],q[5];
cx q[2],q[5];
rz(3.783155349276692) q[5];
cx q[2],q[5];
cx q[4],q[5];
rz(1.0022850213184054) q[5];
h q[5];
rz(1.7958720920408862) q[5];
h q[5];
rz(3*pi) q[5];
cx q[4],q[5];
cx q[6],q[0];
rz(0.30096783466984434) q[0];
h q[0];
rz(1.7958720920408862) q[0];
h q[0];
rz(3*pi) q[0];
cx q[6],q[0];
cx q[2],q[6];
rz(3.6610907194993016) q[6];
cx q[2],q[6];
cx q[7],q[9];
rz(6.559293023378725) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(0.27892972578104924) q[7];
h q[7];
rz(1.7531878769300802) q[7];
h q[7];
cx q[10],q[7];
cx q[8],q[1];
rz(0.3146362491335575) q[1];
h q[1];
rz(1.7531878769300802) q[1];
h q[1];
cx q[8],q[1];
cx q[1],q[3];
rz(10.290437922958748) q[3];
cx q[1],q[3];
cx q[1],q[7];
rz(10.663586108502603) q[7];
cx q[1],q[7];
cx q[3],q[6];
rz(0.7862809510688784) q[6];
h q[6];
rz(1.7958720920408862) q[6];
h q[6];
rz(3*pi) q[6];
cx q[3],q[6];
cx q[8],q[9];
rz(0.2645684025478824) q[9];
cx q[8],q[9];
cx q[10],q[9];
rz(0.2443047511098273) q[9];
h q[9];
rz(1.7531878769300802) q[9];
h q[9];
cx q[10],q[9];
h q[10];
rz(1.7531878769300802) q[10];
h q[10];
cx q[10],q[2];
rz(1.124659529435168) q[2];
h q[2];
rz(1.7958720920408862) q[2];
h q[2];
rz(3*pi) q[2];
cx q[10],q[2];
cx q[11],q[8];
rz(0.25088894671779105) q[8];
h q[8];
rz(1.7531878769300802) q[8];
h q[8];
cx q[11],q[8];
h q[11];
rz(1.7531878769300802) q[11];
h q[11];
cx q[11],q[3];
rz(0.9906833640229307) q[3];
h q[3];
rz(1.7958720920408862) q[3];
h q[3];
rz(3*pi) q[3];
cx q[11],q[3];
cx q[11],q[4];
rz(0.6807874040325448) q[4];
h q[4];
rz(1.7958720920408862) q[4];
h q[4];
rz(3*pi) q[4];
cx q[11],q[4];
cx q[7],q[9];
rz(10.289143017736194) q[9];
cx q[7],q[9];
cx q[10],q[7];
rz(0.9053086828804178) q[7];
h q[7];
rz(1.7958720920408862) q[7];
h q[7];
rz(3*pi) q[7];
cx q[10],q[7];
cx q[8],q[1];
rz(1.4233631020404482) q[1];
h q[1];
rz(1.7958720920408862) q[1];
h q[1];
rz(3*pi) q[1];
cx q[8],q[1];
cx q[8],q[9];
rz(3.8385375343582546) q[9];
cx q[8],q[9];
cx q[10],q[9];
rz(0.4029461043497662) q[9];
h q[9];
rz(1.7958720920408862) q[9];
h q[9];
rz(3*pi) q[9];
cx q[10],q[9];
h q[10];
rz(4.4873132151387) q[10];
h q[10];
cx q[11],q[8];
rz(0.4984740726609136) q[8];
h q[8];
rz(1.7958720920408862) q[8];
h q[8];
rz(3*pi) q[8];
cx q[11],q[8];
h q[11];
rz(4.4873132151387) q[11];
h q[11];
