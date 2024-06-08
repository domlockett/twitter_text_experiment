# Packages
library(dplyr)
library(ggplot2)

# Data
popdata <- read.csv('data/citypopdata.csv')

# User Interface
in1 <- selectInput(
  inputId = 'selected_city',
  label = 'Select a city',
  choices = unique(popdata[['NAME']]))

in2 <- sliderInput(
  inputId = "my_xlims", 
  label = "Set X axis limits",
  min = 2010, 
  max = 2018,
  value = c(2010, 2018))

side <- sidebarPanel('Options', in1, in2)									    
out1 <- textOutput('city_label')
out2 <- tabPanel(
  title = 'Plot',
  plotOutput('city_plot'))

out3 <- tabPanel(
  title = 'Data',
  dataTableOutput('city_table'))

main <- mainPanel(out1, tabsetPanel(out2, out3))

tab1 <- tabPanel(
  title = 'City Population',
  sidebarLayout(side, main))

ui <- navbarPage(
  title = 'Census Population Explorer',
  tab1)

# Server
server <- function(input, output) {
  selected_years <- reactive({
    popdata %>%
      filter(NAME == input[['selected_city']]) %>%
      filter(
        year >= input[['my_xlims']][1],
        year <= input[['my_xlims']][2])
  })
  
  output[['city_label']] <- renderText({
    popdata %>% 
      filter(NAME == input[['selected_city']]) %>%
      slice(1) %>% 
      dplyr::select(NAME, LSAD) %>%
      paste(collapse = ' ')
  })
  
  output[['city_plot']] <- renderPlot({
    ggplot(selected_years(), aes(x = year, y = population)) + 
      geom_line() 
  })
  output[['city_table']] <- renderDataTable({
    selected_years()
  })
}

# Create the Shiny App
shinyApp(ui = ui, server = server)