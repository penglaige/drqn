video_width  = 200
video_height = 160

apple_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>HDRLN</Summary>
              </About>

              <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;1*minecraft:bedrock,3*minecraft:dirt,1*minecraft:grass;2;village"/>
                  <DrawingDecorator>
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

                    <DrawBlock type="fence_gate" x="50" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="49" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="50"/>

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


                    <DrawBlock type="fence_gate" x="50" y="5" z="49" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="48" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="47" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="46" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="45" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="44" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="43" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="42" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="41" face="WEST"/>

                    <DrawBlock type="fence_gate" x="41" y="5" z="49" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="48" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="47" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="46" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="45" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="44" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="43" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="42" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="41" face="WEST"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
                    <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                        <InventoryItem slot="7" type="shears"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <VideoProducer>
                    <Width>''' + str(video_width) + '''</Width>
                    <Height>''' + str(video_height) + '''</Height>
                  </VideoProducer>
                  <ObservationFromGrid>
                    <Grid name="front3x3">
                        <min x="0" y="0" z="0"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="10"/>
                    <Item type="wool" reward="100"/>
                  </RewardForCollectingItem>
                  <DiscreteMovementCommands/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
