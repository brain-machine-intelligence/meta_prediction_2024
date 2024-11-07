% portSpec = FindSerialPort([],1);
portSpec = 'COM5';
joker = ''; sampleFreq = 120; baudRate = 115200; specialSettings = [];
InputBufferSize = sampleFreq * 3600;
readTimeout = max(10*1/sampleFreq, 15);
readTimeout = min(readTimeout, 21);
portSettings = sprintf(['%s %s BaudRate=%i InputBufferSize=%i Terminator=0 ReceiveTimeout=%f ' ...
    'ReceiveLatency=0.0001'], joker, specialSettings, baudRate, InputBufferSize, readTimeout);
myport = IOPort('OpenSerialPort', portSpec, portSettings);
asyncSetup = sprintf('%s BlockingBackgroundRead=1 StartBackgroundRead=1',joker);
IOPort('ConfigureSerialPort', myport, asyncSetup);

[~,~,k,~] = KbCheck;
tic
treceived = 0;
while treceived == 0
    disp("Waiting Sync");
    [pktdata, treceived] = IOPort('Read',myport,1,1);
    pktdata=ceil(pktdata)
    toc
%     disp([pktdata, treceived]);
    [~,~,k,~] = KbCheck;
    tic
end

IOPort('Close',myport)